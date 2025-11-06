import os
from os import path as osp
import torch
import torch.nn.functional as F
import numpy as np
import datetime

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srrs_model import SRRSModel


def split_and_resize(tensor):
    """ 最后没用，直接将 NSS 以 RGB 的分辨率保存了
    输入：tensor，形状为 [B, 6, H, W]
    输出：
        rgb: [B, 3, H, W]
        resized_nss: [B, 3, int(H * 2/3), int(W * 2/3)]
    """
    assert tensor.dim() == 4 and tensor.size(1) == 6, "输入必须为 [B, 6, H, W]"

    rgb = tensor[:, :3, :, :]
    nss = tensor[:, 3:, :, :]

    H, W = nss.shape[2], nss.shape[3]
    new_size = (int(H / 2), int(W / 2))
    resized_nss = F.interpolate(nss, size=new_size, mode='bicubic', align_corners=True)

    return rgb, resized_nss


@MODEL_REGISTRY.register()
class L2SSingleModel(SRRSModel):
    """Base SR model for single rs image super-resolution.
    - Implement amp
    - Change image save way
    - Change metric
    """

    def feed_data(self, data):
        self.sample_path = data["sample_path"]
        self.img_name = data["img_name"]

        self.lq = torch.cat([data["lq"]["rgb"], data["lq"]["nss"]], dim=1).to(self.device)

        if 'gt' in data:
            gt = data['gt']
            gt_rgb = gt["rgb"].to(self.device)

            # 上采样 nss 图像两倍
            gt_nss = gt["nss"].to(self.device)
            gt_nss_up = F.interpolate(gt_nss, scale_factor=2, mode='bicubic')

            # 拼接 RGB 和 NSS
            self.gt = torch.cat([gt_rgb, gt_nss_up], dim=1)

    def log_nan_inf_loss(self, current_iter, loss):
        log_dir = osp.join(self.opt['path']['experiments_root'], "loss", str(current_iter))
        os.makedirs(log_dir, exist_ok=True)

        # 保存元信息
        meta_info = {
            "iter": current_iter,
            "sample_path": self.sample_path,
            "loss": float(loss.detach().cpu()) if torch.isfinite(loss) else str(loss),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loss_type": "NaN" if torch.isnan(loss) else "Inf" if torch.isinf(loss) else "Other"
        }
        torch.save(meta_info, os.path.join(log_dir, f"meta_iter_{current_iter}.pt"))

        # 保存张量数据
        torch.save(self.lq.detach().cpu(), os.path.join(log_dir, f"lq_iter_{current_iter}.pt"))
        torch.save(self.output.detach().cpu(), os.path.join(log_dir, f"out_iter_{current_iter}.pt"))
        torch.save(self.gt.detach().cpu(), os.path.join(log_dir, f"gt_iter_{current_iter}.pt"))

    def _extract_img_name(self, val_data: dict) -> str:
        return val_data['img_name'][0]

    def _save_visuals(self, dataset: str, img_name: str, images: dict[str, np.ndarray]):
        """
        Save RGB and NSS components of input images.

        Args:
            dataset (str): Dataset name.
            img_name (str): Image identifier.
            images (dict): Dictionary containing image arrays, e.g. {"lq": ..., "gt": ..., "sr_100": ...}
        """
        vis_path = self.opt['path']['visualization']
        rgb, nss = {}, {}

        for name, img in images.items():
            if img is None:
                continue
            rgb[name] = img[..., :3]
            nss[name] = img[..., 3:]

        self.rswrite(osp.join(vis_path, "RGB", dataset, img_name), rgb, is_rgb_order=True)
        self.rswrite(osp.join(vis_path, "NSS", dataset, img_name), nss, is_rgb_order=True)
