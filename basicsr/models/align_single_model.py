from os import path as osp
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.models.srrs_l2s_model import L2SSingleModel
from basicsr.metrics import calculate_metric
from basicsr.utils import minusone_one_tensor_to_ubyte_numpy, get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class AlignSingleModel(L2SSingleModel):
    """Base SR model for single rs image super-resolution.
    - Implement amp
    - Change image save way
    - Change metric
    """

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        # Define loss functions
        self.stage1_loss = self.build_optional_loss(train_opt, 'stage1_opt')
        self.stage2_loss = self.build_optional_loss(train_opt, 'stage2_opt')

        # Sanity check
        if self.stage1_loss is None or self.stage2_loss is None:
            raise ValueError("Both 'stage1_opt' and 'stage2_opt' must be specified in the training options.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

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

            lq_up = F.interpolate(self.lq, scale_factor=3, mode='bicubic')
            self.reg_input = torch.cat([lq_up, self.gt], dim=1)

    def _compute_losses(self, stage1_lr, stage2_sr, lq, gt):
        loss_dict = OrderedDict()

        stage1_loss = self.stage1_loss(stage1_lr, lq)
        loss_dict['stage1_loss'] = stage1_loss

        stage2_loss = self.stage2_loss(stage2_sr, gt)
        loss_dict['stage2_loss'] = stage2_loss

        total_loss = stage1_loss + stage2_loss
        return total_loss, loss_dict

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        context = torch.cuda.amp.autocast if self.use_amp else nullcontext

        with context():
            output = self.net_g(self.reg_input)
            stage1_sr, stage2_sr = output['stage1'], output['stage2']

            lq_up = F.interpolate(self.lq, scale_factor=3, mode='bicubic')

            l_total, loss_dict = self._compute_losses(stage1_sr, stage2_sr, lq_up, self.gt)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.use_amp:
            self.amp_scaler.scale(l_total).backward()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()

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
        else:
            detailed_metrics = None

        progress = tqdm(total=len(dataloader), unit='image') if show_progress else None

        for idx, val_data in enumerate(dataloader):
            img_name = self._extract_img_name(val_data)
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # input visuals: [-1,1] tensor b c h w
            # output image: [0,255] numpy h w c
            converted_visuals = {}
            for name, tensor in visuals.items():
                if tensor is not None:
                    converted_visuals[name] = minusone_one_tensor_to_ubyte_numpy(tensor)

            self._release_gpu_memory()

            if metrics_enabled:
                self._compute_registration_metrics(img_name,
                                                   sr_stage1=converted_visuals["sr_stage1"],
                                                   sr_stage2=converted_visuals["sr_stage2"],
                                                   lq_up_img=converted_visuals["lq_up"],
                                                   gt_img=converted_visuals["gt"],
                                                   df=detailed_metrics)

            if save_img:
                # 重命名所有以 "sr" 开头的键，加上当前迭代号
                for key in list(converted_visuals.keys()):
                    if key.startswith("sr"):
                        new_key = f"{key}_{current_iter}"
                        converted_visuals[new_key] = converted_visuals.pop(key)
                self._save_visuals(dataset_name, img_name, converted_visuals)

            if progress:
                progress.update(1)
                progress.set_description(f'Test {img_name}')

        if progress:
            progress.close()

        if metrics_enabled:
            self._finalize_metrics(idx + 1, dataset_name, current_iter, tb_logger)
            self._save_metrics_csv(dataset_name, current_iter, detailed_metrics)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.reg_input)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.reg_input)
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq_up'] = F.interpolate(self.lq.detach().cpu(), scale_factor=3, mode='bicubic')

        out_dict['sr_stage1'] = self.output["stage1"].detach().cpu()

        out_dict['sr_stage2'] = self.output["stage2"].detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()

        return out_dict

    def _prepare_metrics(self, dataset_name: str):
        metrics = self.opt['val']['metrics']
        if not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {}
            for name in metrics:
                self.metric_results[f"{name}_stage1"] = 0.0
                self.metric_results[f"{name}_stage2"] = 0.0
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        self.metric_results = {metric: 0 for metric in self.metric_results}

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
            record[f"{metric}_stage1"] = dict(better=better, val=init_val, iter=-1)
            record[f"{metric}_stage2"] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _compute_registration_metrics(self, img_name: str, sr_stage1: np.ndarray, sr_stage2: np.ndarray,
                                      lq_up_img: np.ndarray, gt_img: np.ndarray | None, df: pd.DataFrame):

        # print("sr_stage1_down:", sr_stage1_down.shape)
        # print("sr_stage2:", sr_stage2.shape)
        # print("lq_img:", lq_img.shape)
        # print("gt_img:", gt_img.shape)

        metric_stage1_data = {'img': sr_stage1, 'img2': lq_up_img}
        metric_stage2_data = {'img': sr_stage2, 'img2': gt_img}

        for name, opt in self.opt['val']['metrics'].items():
            stage1_score = calculate_metric(metric_stage1_data, opt)
            self.metric_results[f"{name}_stage1"] += stage1_score
            df.loc[img_name, f"{name}_stage1"] = stage1_score

            stage2_score = calculate_metric(metric_stage2_data, opt)
            self.metric_results[f"{name}_stage2"] += stage2_score
            df.loc[img_name, f"{name}_stage2"] = stage2_score

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
