from os import path as osp
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.utils import minusone_one_tensor_to_ubyte_numpy
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srrs_model import SRRSModel


@MODEL_REGISTRY.register()
class L2SSingleHMModel(SRRSModel):
    """Base SR model for single rs image super-resolution.
    - Implement amp
    - Change image save way
    - Change metric
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.hm_weight = self.opt["hm_loss_weight"]
        self.gt_weight = self.opt["gt_loss_weight"]

    def feed_data(self, data):
        # 低分辨率输入：Landsat RGB + NSS
        self.lq = torch.cat([data["lq"]["rgb"], data["lq"]["nss"]], dim=1).to(self.device)

        # 高分 Ground Truth
        if 'gt' in data:
            gt_rgb = data['gt']["rgb"].to(self.device)
            gt_nss = data['gt']["nss"].to(self.device)
            gt_nss_up = F.interpolate(gt_nss, scale_factor=2, mode='bicubic')
            self.gt = torch.cat([gt_rgb, gt_nss_up], dim=1)

        # 高分 Histogram-Matched 图像（与 gt 完全一致结构）
        if 'hm' in data:
            hm_rgb = data['hm']["rgb"].to(self.device)
            hm_nss = data['hm']["nss"].to(self.device)
            hm_nss_up = F.interpolate(hm_nss, scale_factor=2, mode='bicubic')
            self.hm = torch.cat([hm_rgb, hm_nss_up], dim=1)

    def _compute_hm_loss(self, output, target):
        loss = 0
        loss_dict = {}

        if self.cri_pix:
            l_pix = self.cri_pix(output, target)
            loss += l_pix
            loss_dict['l_pix_hm'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(output, target)
            if l_percep is not None:
                loss += l_percep
                loss_dict['l_percep_hm'] = l_percep
            if l_style is not None:
                loss += l_style
                loss_dict['l_style_hm'] = l_style

        return loss, loss_dict

    def _compute_gt_loss(self, output, target):
        loss = 0
        loss_dict = {}

        if self.cri_pix:
            l_pix = self.cri_pix(output, target)
            loss += l_pix
            loss_dict['l_pix_gt'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(output, target)
            if l_percep is not None:
                loss += l_percep
                loss_dict['l_percep_gt'] = l_percep
            if l_style is not None:
                loss += l_style
                loss_dict['l_style_gt'] = l_style

        return loss, loss_dict

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        context = torch.cuda.amp.autocast if self.use_amp else nullcontext

        with context():
            result = self.net_g(self.lq)
            self.output_hm, self.output_gt = result['hm'], result['gt']

            # 分别计算 HM 和 GT 的损失
            loss_hm, loss_dict_hm = self._compute_hm_loss(self.output_hm, self.hm)
            loss_gt, loss_dict_gt = self._compute_gt_loss(self.output_gt, self.gt)

            # 权重合并总损失
            l_total = self.hm_weight * loss_hm + self.gt_weight * loss_gt

            # 合并日志
            loss_dict = {**loss_dict_hm, **loss_dict_gt}

        self.safe_backward_and_step(l_total)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter: int, tb_logger, save_img: bool):
        dataset_name = dataloader.dataset.opt['name']
        val_opt = self.opt['val']
        metrics_enabled = val_opt.get('metrics') is not None
        show_progress = val_opt.get('pbar', False)

        if metrics_enabled:
            self._prepare_metrics(dataset_name)
            detailed_metrics = pd.DataFrame()

        progress = tqdm(total=len(dataloader), unit='image') if show_progress else None

        for idx, val_data in enumerate(dataloader):
            img_name = self._extract_img_name(val_data)
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # input visuals: [-1,1] tensor b c h w
            # output image: [0,255] numpy h w c
            lq_img = minusone_one_tensor_to_ubyte_numpy(visuals['lq'])

            sr_hm_img = minusone_one_tensor_to_ubyte_numpy(visuals['result_hm'])
            sr_gt_img = minusone_one_tensor_to_ubyte_numpy(visuals['result_gt'])

            hm_img = minusone_one_tensor_to_ubyte_numpy(visuals['hm']) if 'hm' in visuals else None
            gt_img = minusone_one_tensor_to_ubyte_numpy(visuals['gt']) if 'gt' in visuals else None

            self._release_gpu_memory()

            if metrics_enabled:
                self._compute_hm_metrics(img_name, sr_hm_img=sr_hm_img, sr_gt_img=sr_gt_img,
                                         hm_img=hm_img, gt_img=gt_img, df=detailed_metrics)

            if save_img:
                self._save_hm_visuals(dataset_name, img_name, current_iter, lq_img, sr_hm_img, sr_gt_img,
                                      hm_img, gt_img)

            if progress:
                progress.update(1)
                progress.set_description(f'Test {img_name}')

        if progress:
            progress.close()

        if metrics_enabled:
            self._finalize_metrics(idx + 1, dataset_name, current_iter, tb_logger)
            self._save_metrics_csv(dataset_name, current_iter, detailed_metrics)

    def _extract_img_name(self, val_data: dict) -> str:
        return val_data['img_name'][0]

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[f"{metric}_hm"] = dict(better=better, val=init_val, iter=-1)
            record[f"{metric}_gt"] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _prepare_metrics(self, dataset_name: str):
        metrics = self.opt['val']['metrics']
        if not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {}
            for name in metrics:
                self.metric_results[f"{name}_hm"] = 0.0
                self.metric_results[f"{name}_gt"] = 0.0
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        self.metric_results = {metric: 0 for metric in self.metric_results}

    def _compute_hm_metrics(self, img_name: str, sr_hm_img: np.ndarray, sr_gt_img: np.ndarray,
                            hm_img: np.ndarray | None, gt_img: np.ndarray | None, df: pd.DataFrame):
        if gt_img is None and hm_img is None:
            return

        metric_hm_data = {'img': sr_hm_img, 'img2': hm_img}
        metric_gt_data = {'img': sr_gt_img, 'img2': gt_img}

        for name, opt in self.opt['val']['metrics'].items():
            hm_score = calculate_metric(metric_hm_data, opt)
            self.metric_results[f"{name}_hm"] += hm_score
            df.loc[img_name, f"{name}_hm"] = hm_score

            gt_score = calculate_metric(metric_gt_data, opt)
            self.metric_results[f"{name}_gt"] += gt_score
            df.loc[img_name, f"{name}_gt"] = gt_score

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()

        out_dict['result_gt'] = self.output["gt"].detach().cpu()
        out_dict['result_hm'] = self.output["hm"].detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'hm'):
            out_dict['hm'] = self.hm.detach().cpu()

        return out_dict

    def _save_hm_visuals(self, dataset: str, img_name: str, iter_num: int,
                         lq: np.ndarray, sr_hm: np.ndarray, sr_gt: np.ndarray,
                         hm: np.ndarray | None, gt: np.ndarray | None):

        vis_path = self.opt['path']['visualization']
        rgb = {
            "lq": lq[..., :3],
            f"sr_{iter_num}_hm": sr_hm[..., :3],
            f"sr_{iter_num}_gt": sr_gt[..., :3],
            "hm": hm[..., :3] if hm is not None else None,
            "gt": gt[..., :3] if gt is not None else None,
        }
        nss = {
            "lq": lq[..., 3:],
            f"sr_{iter_num}_hm": sr_hm[..., 3:],
            f"sr_{iter_num}_gt": sr_gt[..., 3:],
            "hm": hm[..., 3:] if hm is not None else None,
            "gt": gt[..., 3:] if gt is not None else None,
        }

        self.rswrite(osp.join(vis_path, "RGB", dataset, img_name), rgb, is_rgb_order=True)
        self.rswrite(osp.join(vis_path, "NSS", dataset, img_name), nss, is_rgb_order=True)