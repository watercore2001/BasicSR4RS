import os
import torch
from torch.utils import data

from basicsr.data.l2s_single_dataset import (merge_last_three_folder_names, load_grouped_numpy,
                                             filter_metrics, get_sample_current_paths)
from basicsr.data.transforms import (augment, paired_random_crop, paired_central_crop,
                                     chw2hwc, resize_hwc, LandsatNorm, SentinelNorm)
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

BAND_NUM = 3
RGB_SCALE = 0.3
NSS_SCALE = 0.5


class L2SSingleHMDataset(data.Dataset):
    sources = ['landsat', 'sentinel']
    rgb_scale = 3
    nss_scale = 1.5
    rgb_bands = ['red.tif', 'green.tif', 'blue.tif']
    nss_bands = ['nir08.tif', 'swir16.tif', 'swir22.tif']

    """
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            root_path: str
            phase: Literal["train", "val"]
            psnr_thresh: float
            ssim_thresh: float
            gt_size: int. RGB HR image size.
            use_hflip: bool. Whether to horizontally flip the image.
            use_rot: bool. Whether to rotate 90angle the image.
            bands: list[str]. All bands are ["red", "green", "blue", "nir08", "swir16", "swir22"].
    """

    def __init__(self, opt: dict):
        self.opt = opt
        self.root_path = opt['root_path']

        self.metric_dict = filter_metrics(self.root_path, psnr_min=opt["psnr_min"], ssim_min=opt["ssim_min"],
                                          psnr_max=opt["psnr_max"])
        self.sample_current_paths = get_sample_current_paths(self.metric_dict)

        self.landsat_rgb_norm = LandsatNorm(BAND_NUM, RGB_SCALE)
        self.landsat_nss_norm = LandsatNorm(BAND_NUM, NSS_SCALE)
        self.sentinel_rgb_norm = SentinelNorm(BAND_NUM, RGB_SCALE)
        self.sentinel_nss_norm = SentinelNorm(BAND_NUM, NSS_SCALE)


    def get_sample_path(self, idx: int):
        return os.path.join(self.root_path, str(self.sample_current_paths[idx]))

    def __len__(self):
        return len(self.sample_current_paths)

    def __getitem__(self, idx):
        sample_path = self.get_sample_path(idx)

        # numpy hwc rgb
        lq_rgb_numpy = chw2hwc(load_grouped_numpy(sample_path, "landsat", self.rgb_bands))
        lq_nss_numpy = chw2hwc(load_grouped_numpy(sample_path, "landsat", self.nss_bands))

        gt_rgb_numpy = chw2hwc(load_grouped_numpy(sample_path, "sentinel", self.rgb_bands))
        gt_nss_numpy = chw2hwc(load_grouped_numpy(sample_path, "sentinel", self.nss_bands))
        gt_nss_numpy_upscale = resize_hwc(gt_nss_numpy, self.rgb_scale / self.nss_scale)

        hm_rgb_numpy = chw2hwc(load_grouped_numpy(sample_path, "sentinel_hm", self.rgb_bands))
        hm_nss_numpy = chw2hwc(load_grouped_numpy(sample_path, "sentinel_hm", self.nss_bands))
        hm_nss_numpy_upscale = resize_hwc(hm_nss_numpy, self.rgb_scale / self.nss_scale)

        rgb_gt_size = self.opt['gt_size']

        img_gts = [gt_rgb_numpy, gt_nss_numpy_upscale, hm_rgb_numpy, hm_nss_numpy_upscale]
        img_lqs = [lq_rgb_numpy, lq_nss_numpy]

        # augmentation for train
        if self.opt['phase'] == 'train':

            # random crop
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs,
                                                  rgb_gt_size, self.rgb_scale, sample_path)

            # flip, rotation
            all_image = augment(img_gts + img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
            img_gts, img_lqs = all_image[:4], all_image[4:]

        else:
            # central crop
            img_gts, img_lqs = paired_central_crop(img_gts, img_lqs, rgb_gt_size, self.rgb_scale, sample_path)

        gt_rgb_numpy, gt_nss_numpy_upscale, hm_rgb_numpy, hm_nss_numpy_upscale = img_gts
        gt_nss_numpy = resize_hwc(gt_nss_numpy_upscale, self.nss_scale / self.rgb_scale)
        hm_nss_numpy = resize_hwc(hm_nss_numpy_upscale, self.nss_scale / self.rgb_scale)

        lq_rgb_numpy, lq_nss_numpy = img_lqs

        # Order: CHW rgb/nss
        gt_rgb_tensor, gt_nss_tensor, hm_rgb_tensor, hm_nss_tensor, lq_rgb_tensor, lq_nss_tensor = img2tensor(
            [gt_rgb_numpy, gt_nss_numpy, hm_rgb_numpy, hm_nss_numpy, lq_rgb_numpy, lq_nss_numpy], bgr2rgb=False)

        # norm to [-1, 1]
        gt_rgb_tensor = self.sentinel_rgb_norm(gt_rgb_tensor)
        gt_nss_tensor = self.sentinel_nss_norm(gt_nss_tensor)
        hm_rgb_tensor = self.landsat_rgb_norm(hm_rgb_tensor)
        hm_nss_tensor = self.landsat_nss_norm(hm_nss_tensor)
        lq_rgb_tensor = self.landsat_rgb_norm(lq_rgb_tensor)
        lq_nss_tensor = self.landsat_nss_norm(lq_nss_tensor)

        result = {
            "lq": {"rgb": lq_rgb_tensor, "nss": lq_nss_tensor},
            "gt": {"rgb": gt_rgb_tensor, "nss": gt_nss_tensor},
            "hm": {"rgb": hm_rgb_tensor, "nss": hm_nss_tensor},
            "sample_path": sample_path,
            "img_name": merge_last_three_folder_names(sample_path),
        }

        # 🔍 检查所有张量是否存在 NaN
        for key1 in ["lq", "gt", "hm"]:
            for key2 in ["rgb", "nss"]:
                tensor = result[key1][key2]
                if torch.isnan(tensor).any():
                    raise ValueError(f"检测到 NaN：{key1}/{key2}，样本路径：{sample_path}")

        return result


@DATASET_REGISTRY.register()
class L2SSingleHMSplitDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        split_percent = self.opt['split_percent']
        overall_dataset = L2SSingleHMDataset(opt)
        datasets = data.random_split(overall_dataset, split_percent, torch.Generator().manual_seed(0))

        split = opt['split']
        self.dataset = datasets[split]

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
