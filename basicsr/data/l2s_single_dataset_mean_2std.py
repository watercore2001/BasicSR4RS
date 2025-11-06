import json
import os
from typing import Literal

import numpy as np
import rasterio

from torch.utils import data

from basicsr.data.transforms import (augment, paired_random_crop, paired_central_crop,
                                     chw2hwc, resize_hwc, build_normalizer)
from basicsr.utils import img2tensor


def load_grouped_numpy(window_path: str, source: Literal["landsat", "sentinel"], band_list: list[str]):
    source_path = os.path.join(window_path, source)
    band_arrays = []

    for band in band_list:
        band_path = os.path.join(source_path, band)
        with rasterio.open(band_path) as src:
            array = src.read(1).astype(np.float32)
            array[array == 0] = np.nan
            band_arrays.append(array)

    stacked = np.stack(band_arrays)  # shape: [C, H, W]
    return stacked


class L2SSingleDataset(data.Dataset):
    sources = ['landsat', 'sentinel']
    rgb_scale = 3
    nss_scale = 1.5
    rgb_bands = ['red.tif', 'green.tif', 'blue.tif']
    nss_bands = ['nir08.tif', 'swir16.tif', 'swir22.tif']
    landsat_mean_std = {
        "blue": {
            "mean": 9563.7984,
            "std": 2961.4631
        },
        "green": {
            "mean": 10615.2811,
            "std": 2908.3771
        },
        "red": {
            "mean": 10721.1473,
            "std": 3259.4143
        },
        "nir08": {
            "mean": 14673.4545,
            "std": 4900.9225
        },
        "swir16": {
            "mean": 13539.8734,
            "std": 3882.9049
        },
        "swir22": {
            "mean": 11925.1374,
            "std": 3245.9576
        },
        "thumbnail": {
            "mean": 199.0643,
            "std": 49.021
        },
    }
    sentinel_mean_std = {
        "blue": {
            "mean": 795.0896,
            "std": 789.0347
        },
        "green": {
            "mean": 985.7132,
            "std": 759.0325
        },
        "red": {
            "mean": 1035.7568,
            "std": 842.9308
        },
        "nir08": {
            "mean": 2212.2311,
            "std": 1290.3774
        },
        "swir16": {
            "mean": 1912.4766,
            "std": 1082.7832
        },
        "swir22": {
            "mean": 1444.6185,
            "std": 942.3633
        },
        "thumbnail": {
            "mean": 121.2004,
            "std": 79.3277
        },
        "nir": {
            "mean": 2196.3676,
            "std": 1288.2695
        }
    }

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

        self.metric_dict = self.filter_metrics(self.root_path, opt["psnr_thresh"], opt["ssim_thresh"])
        self.sample_current_paths = self.get_sample_current_paths(self.metric_dict)

        self.landsat_rgb_norm = build_normalizer(self.landsat_mean_std, ["red", "green", "blue"])
        self.landsat_nss_norm = build_normalizer(self.landsat_mean_std, ["nir08", "swir16", "swir22"])
        self.sentinel_rgb_norm = build_normalizer(self.sentinel_mean_std, ["red", "green", "blue"])
        self.sentinel_nss_norm = build_normalizer(self.sentinel_mean_std, ["nir08", "swir16", "swir22"])

    @staticmethod
    def filter_metrics(root_path: str, psnr_thresh: float, ssim_thresh: float):
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
        filtered_result = {}

        for tile_id in os.listdir(root_path):
            tile_path = os.path.join(root_path, tile_id)
            if not os.path.isdir(tile_path):
                continue

            metrics_path = os.path.join(tile_path, "metric.json")
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
                    psnr = values.get("psnr", 0)
                    ssim = values.get("ssim", 0)

                    if psnr >= psnr_thresh and ssim >= ssim_thresh:
                        if window_id not in one_tile_metrics:
                            one_tile_metrics[window_id] = {}
                            one_tile_metrics[window_id][time_str] = {
                                "psnr": psnr,
                                "ssim": ssim
                            }

            if one_tile_metrics:
                filtered_result[tile_id] = one_tile_metrics

        return filtered_result

    @staticmethod
    def get_sample_current_paths(metric_dict):
        results = []
        for tile_id, tile_dict in metric_dict.items():
            for window_id, time_dict in tile_dict.items():
                current_paths = [os.path.join(tile_id, window_id, time_id) for time_id in time_dict.keys()]
                results.extend(current_paths)

        return results

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
            "landsat": {"rgb": lq_rgb_tensor, "nss": lq_nss_tensor},
            "sentinel": {"rgb": gt_rgb_tensor, "nss": gt_nss_tensor},
            "window_path": sample_path
        }

        return result


