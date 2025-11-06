import json
import os
from typing import Literal
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch.utils import data

from basicsr.data.transforms import (augment, paired_random_crop, paired_central_crop,
                                     chw2hwc, resize_hwc, LandsatNorm, SentinelNorm)
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

BAND_NUM = 3
RGB_SCALE = 0.3
NSS_SCALE = 0.5

def merge_last_three_folder_names(folder_path: str, sep="_") -> str:
    """
    输入路径字符串，返回最后三级文件夹名合并后的字符串
    """
    path = Path(folder_path).resolve()
    parts = path.parts

    if len(parts) < 3:
        raise ValueError("路径层级不足三层")

    last_three = parts[-3:]
    return sep.join(last_three)


def load_grouped_numpy(window_path: str, source: Literal["landsat", "sentinel", "sentinel_hm"], band_list: list[str]):
    source_path = os.path.join(window_path, source)
    band_arrays = []

    for band in band_list:
        band_path = os.path.join(source_path, band)
        with rasterio.open(band_path) as src:
            array = src.read(1).astype(np.float32)

            # 数据集中出现的0并不是nodata, 而是精度处理中数值很小的数，保留就好了
            # if (array == 0).any():
            #     raise ValueError(f"发现波段 {band} 中包含 0 值像素，位置：{band_path}")

            band_arrays.append(array)

    stacked = np.stack(band_arrays)  # shape: [C, H, W]
    return stacked


def filter_metrics(root_path: str, psnr_min: float, ssim_min: float, psnr_max: float, use_hm = True):
    """
    return: {
               "tile_id": {
                    "window_id": {
                        "time": {
                            "psnr": ...,
                            "ssim": ...
                        }
                    }
               }
            }
    """
    if use_hm:
        metric_filename = "metric_hm.json"
    else:
        metric_filename = "metric.json"

    filtered_result = {}
    total_samples = 0
    qualified_samples = 0

    for tile_id in os.listdir(root_path):
        tile_path = os.path.join(root_path, tile_id)
        if not os.path.isdir(tile_path):
            continue

        metrics_path = os.path.join(tile_path, metric_filename)
        if not os.path.exists(metrics_path):
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"跳过损坏的 JSON 文件：{metrics_path}，错误：{e}")
            continue

        one_tile_metrics = {}

        for window_id, time_dict in metrics.items():
            for time_str, values in time_dict.items():
                total_samples += 1
                psnr = values.get("psnr", 0)
                ssim = values.get("ssim", 0)

                if psnr_max >= psnr >= psnr_min and ssim >= ssim_min:
                    qualified_samples += 1

                    if window_id not in one_tile_metrics:
                        one_tile_metrics[window_id] = {}

                    one_tile_metrics[window_id][time_str] = {
                        "psnr": psnr,
                        "ssim": ssim
                    }

        if one_tile_metrics:
            filtered_result[tile_id] = one_tile_metrics

    print(f"总样本数：{total_samples}")
    print(f"合格样本数：{qualified_samples}")

    return filtered_result

def get_sample_current_paths(metric_dict):
    results = []
    for tile_id, tile_dict in metric_dict.items():
        for window_id, time_dict in tile_dict.items():
            current_paths = [os.path.join(tile_id, window_id, time_id) for time_id in time_dict.keys()]
            results.extend(current_paths)

    return results

class L2SSingleDataset(data.Dataset):
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

        # random crop
        rgb_gt_size = self.opt['gt_size']

        img_gts = [gt_rgb_numpy, gt_nss_numpy_upscale]
        img_lqs = [lq_rgb_numpy, lq_nss_numpy]

        # augmentation for train
        if self.opt['phase'] == 'train':

            # random crop
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs,
                                                  rgb_gt_size, self.rgb_scale, sample_path)

            # flip, rotation
            all_image = augment(img_gts + img_lqs, self.opt['use_hflip'], self.opt['use_rot'])
            img_gts, img_lqs = all_image[:2], all_image[2:]

        else:
            # central crop
            img_gts, img_lqs = paired_central_crop(img_gts, img_lqs, rgb_gt_size, self.rgb_scale, sample_path)

        gt_rgb_numpy, gt_nss_numpy_upscale = img_gts
        gt_nss_numpy = resize_hwc(gt_nss_numpy_upscale, self.nss_scale / self.rgb_scale)

        lq_rgb_numpy, lq_nss_numpy = img_lqs

        # Order: CHW rgb/nss
        gt_rgb_tensor, gt_nss_tensor, lq_rgb_tensor, lq_nss_tensor = img2tensor(
            [gt_rgb_numpy, gt_nss_numpy, lq_rgb_numpy, lq_nss_numpy], bgr2rgb=False)

        gt_rgb_tensor = self.sentinel_rgb_norm(gt_rgb_tensor)
        gt_nss_tensor = self.sentinel_nss_norm(gt_nss_tensor)
        lq_rgb_tensor = self.landsat_rgb_norm(lq_rgb_tensor)
        lq_nss_tensor = self.landsat_nss_norm(lq_nss_tensor)

        result = {
            "lq": {"rgb": lq_rgb_tensor, "nss": lq_nss_tensor},
            "gt": {"rgb": gt_rgb_tensor, "nss": gt_nss_tensor},
            "sample_path": sample_path,
            "img_name": merge_last_three_folder_names(sample_path),
        }

        return result


@DATASET_REGISTRY.register()
class L2SSingleSplitDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        split_percent = self.opt['split_percent']
        overall_dataset = L2SSingleDataset(opt)
        datasets = data.random_split(overall_dataset, split_percent, torch.Generator().manual_seed(0))

        split = opt['split']
        self.dataset = datasets[split]

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
