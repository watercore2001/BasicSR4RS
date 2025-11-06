import torch
import rasterio as rio
import numpy as np
import tacoreader
from tacoreader import TortillaDataFrame
from torch.utils import data
from torchvision.transforms import Normalize

from basicsr.data.transforms import augment, paired_random_crop, paired_central_crop
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


class TacoDataset(data.Dataset):
    """

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            taco_paths: str | list[str]. Path to taco files
            scale: int. Scale factor
            phase: Literal["train", "val"]
            gt_size: int. HR image size.
            use_hflip: bool. Whether to horizontally flip the image.
            use_rot: bool. Whether to rotate 90angle the image.
            bands: list[int]
    """
    def __init__(self, opt: dict):
        self.opt = opt
        self.scale = opt['scale']

        self.band_idx = opt['band_idx']
        self.scale_max = 3000
        self.norm = Normalize(mean=[0.5]*len(self.band_idx), std=[0.5] * len(self.band_idx))

        # Load the dataset once in memory
        self.dataset: TortillaDataFrame = tacoreader.load(opt["taco_paths"])
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample: TortillaDataFrame = self.dataset.read(idx)
        lq_path: str = sample.read(0)
        gt_path: str = sample.read(1)

        # Load lr and hr images. Dimension Order: CHW. Channel Order: RGB NIR.
        with rio.open(lq_path) as src, rio.open(gt_path) as dst:
            img_lq: np.ndarray = src.read(self.band_idx)
            img_gt: np.ndarray = dst.read(self.band_idx)

        # change dimension to HWC.
        # ascontiguousarray() for opencv.
        img_lq = np.ascontiguousarray(img_lq.transpose(1, 2, 0))
        img_gt = np.ascontiguousarray(img_gt.transpose(1, 2, 0))

        # random crop
        gt_size = self.opt['gt_size']

        # augmentation for train
        if self.opt['phase'] == 'train':
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        else:
            # central crop
            img_gt, img_lq = paired_central_crop(img_gt, img_lq, gt_size, self.scale, gt_path)

        # numpy to tensor, HWC -> CHW.
        # Default Dimension order in DL is BCHW
        # Default Channel order in DL is RGB. So img_gt has been RGB+NIR order
        img_gt = img_gt.astype(np.float32)
        img_lq = img_lq.astype(np.float32)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        # normalized [-1, 1]
        img_lq = self.norm(img_lq / self.scale_max)
        img_gt = self.norm(img_gt / self.scale_max)

        # B RGB+NIR H W
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'band_idx': self.band_idx}


@DATASET_REGISTRY.register()
class TacoSplitDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        split_percent = self.opt['split_percent']
        overall_dataset = TacoDataset(opt)
        datasets = data.random_split(overall_dataset, split_percent, torch.Generator().manual_seed(0))

        split = opt['split']
        self.dataset = datasets[split]

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)
