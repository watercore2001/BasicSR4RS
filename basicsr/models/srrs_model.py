from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from contextlib import nullcontext
import pandas as pd
import cv2
import torch
import numpy as np

from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, minusone_one_tensor_to_ubyte_numpy
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SRRSModel(SRModel):
    """Base SR model for single rs image super-resolution.
    - Implement amp
    - Change image save way
    - Change metric
    """

    def setup_optimizers(self):
        super().setup_optimizers()

        # amp settings
        self.use_amp = self.opt["train"]["use_amp"]
        self.amp_scaler = None
        if self.use_amp:
            self.amp_scaler = torch.cuda.amp.GradScaler()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # 前向传播
        context = torch.cuda.amp.autocast if self.use_amp else nullcontext
        with context():
            # 不应该将反向传播放进来
            # autocast 的主要作用是：在 前向传播阶段 自动将某些操作转换为混合精度（float16），以减少显存占用并加速计算。
            # 它不会影响反向传播（backward()）或优化器更新（step()）的行为。
            self.output = self.net_g(self.lq)
            l_total = 0
            loss_dict = OrderedDict()

            # pixel loss
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 反向传播
        if torch.isnan(l_total) or torch.isinf(l_total):
            print("Loss is NaN or Inf. Skipping optimizer step.")
            self.log_nan_inf_loss(current_iter, l_total)

            # 释放优化器
            self.optimizer_g.zero_grad()
            # 释放计算图
            del self.lq
            del self.gt
            del self.output
            # 可选：释放未使用显存
            torch.cuda.empty_cache()
            return

        if self.use_amp:
            self.amp_scaler.scale(l_total).backward()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def log_nan_inf_loss(self, current_iter, loss):
        pass

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

            visuals = self.get_current_visuals(current_iter)
            # input visuals: [-1,1] tensor b c h w
            # output image: [0,255] numpy h w c
            # 转换所有图像为 numpy 格式
            converted_visuals = {}
            for name, tensor in visuals.items():
                if tensor is not None:
                    converted_visuals[name] = minusone_one_tensor_to_ubyte_numpy(tensor)

            self._release_gpu_memory()

            # 评估指标（仍然只对 sr 和 gt）
            if metrics_enabled and 'sr' in converted_visuals and 'gt' in converted_visuals:
                self._compute_metrics(img_name, converted_visuals['sr'], converted_visuals['gt'], detailed_metrics)
                converted_visuals.pop("sr") # sr 只用于评估指标，不保存

            if save_img:
                self._save_visuals(dataset_name, img_name, converted_visuals)

            if progress:
                progress.update(1)
                progress.set_description(f'Test {img_name}')

        if progress:
            progress.close()

        if metrics_enabled:
            self._finalize_metrics(idx + 1, dataset_name, current_iter, tb_logger)
            self._save_metrics_csv(dataset_name, current_iter, detailed_metrics)

    def _prepare_metrics(self, dataset_name: str):
        metrics = self.opt['val']['metrics']

        # 首次初始化 metric_results
        if not hasattr(self, 'metric_results'):
            self.metric_results = {name: 0.0 for name in metrics}

        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        self.metric_results = {metric: 0 for metric in self.metric_results}

    def _extract_img_name(self, val_data: dict) -> str:
        lq_path = val_data['lq_path'][0]
        return osp.basename(lq_path.split(',')[0]) if lq_path.endswith('.taco') else osp.splitext(lq_path)[0]

    def _release_gpu_memory(self):
        for attr in ('lq', 'output', 'gt'):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()

    def _compute_metrics(self, img_name: str, sr_img: np.ndarray, gt_img: np.ndarray | None, df: pd.DataFrame):
        if gt_img is None:
            return

        metric_data = {'img': sr_img, 'img2': gt_img}

        # aggregate_prefixes = ['niqe_', 'lpips_']
        # aggregated_scores = {prefix: [] for prefix in aggregate_prefixes}
        individual_scores = {}

        # 先计算所有指标并收集聚合项
        for name, opt in self.opt['val']['metrics'].items():
            score = calculate_metric(metric_data, opt)
            individual_scores[name] = score

            # for prefix in aggregate_prefixes:
            #     if name.startswith(prefix):
            #         aggregated_scores[prefix].append(score)

        # 对每个指定前缀进行平均聚合
        # for prefix, scores in aggregated_scores.items():
        #     if scores:
        #         avg_score = sum(scores) / len(scores)
        #         prefix_name = prefix.rstrip('_')
        #         individual_scores[prefix_name] = avg_score

        # 写入所有指标（包括聚合项）到 df 和 metric_results
        for name, score in individual_scores.items():
            df.loc[img_name, name] = score
            self.metric_results[name] += score

    def _save_visuals(self, dataset: str, img_name: str, images: dict[str, np.ndarray]):
        """
        Save RGB and NIR components of input images.

        Args:
            dataset (str): Dataset name.
            img_name (str): Image identifier.
            images (dict): Dictionary containing image arrays, e.g. {"lq": ..., "gt": ..., "sr_100": ...}
        """
        vis_path = self.opt['path']['visualization']
        rgb, nir = {}, {}

        for name, img in images.items():
            if img is None:
                continue
            rgb[name] = img[..., :3]
            nir[name] = img[..., [3]]

        self.rswrite(osp.join(vis_path, "RGB", dataset, img_name), rgb, is_rgb_order=True)
        self.rswrite(osp.join(vis_path, "NIR", dataset, img_name), nir, is_rgb_order=False)

    def _save_metrics_csv(self, dataset: str, iter_num: int, df: pd.DataFrame):
        csv_path = osp.join(self.opt['path']['visualization'], f"{dataset}_{iter_num}.csv")
        df.to_csv(csv_path)

    def _finalize_metrics(self, total: int, dataset: str, iter_num: int, tb_logger):
        for name in self.metric_results:
            self.metric_results[name] /= total
            self._update_best_metric_result(dataset, name, self.metric_results[name], iter_num)
        self._log_validation_metric_values(iter_num, dataset, tb_logger)

    def rswrite(self, folder: str, filename_dict: dict, is_rgb_order: bool):
        """
        Input shape: HWC
        """
        for key, value in filename_dict.items():
            save_file_path = osp.join(folder, f'{key}.png')
            if not osp.exists(save_file_path) and value is not None:
                img_to_save = cv2.cvtColor(value, cv2.COLOR_RGB2BGR) if is_rgb_order else value
                imwrite(img_to_save, save_file_path)

        # if not self.opt["is_train"]:
        #     # save all image in one picture
        #     all_path = osp.join(folder, dataset_name, f'{img_name}_{suffix}.png')
        #     if gt_img is None:
        #         save_lq_sr_image(lq_img, sr_img, all_path)
        #     else:
        #         save_all_image(lq_img, sr_img, gt_img, all_path)

    def get_training_state(self, epoch, current_iter):
        state = super().get_training_state(epoch, current_iter)
        if self.amp_scaler is not None:
            state['amp_scaler'] = self.amp_scaler.state_dict()
        return state

    def resume_training(self, resume_state):
        super().resume_training(resume_state)

        if self.amp_scaler is not None:
            if "amp_scaler" in resume_state:
                self.amp_scaler.load_state_dict(resume_state["amp_scaler"])
