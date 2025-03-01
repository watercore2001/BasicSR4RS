import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from contextlib import nullcontext
import pandas as pd

import torch

from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2ubyte_image, save_lq_sr_image, save_all_image
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SRRSModel(SRModel):
    """Base SR model for single rs image super-resolution.
    - Implement amp
    - Change image save way
    - Change metric
    """

    def __init__(self, opt):
        self.use_amp = opt["train"]["use_amp"]
        self.amp_scaler = None
        super().__init__(opt)

    def setup_optimizers(self):
        super().setup_optimizers()

        # amp settings
        if self.use_amp:
            self.amp_scaler = torch.cuda.amp.GradScaler()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        context = torch.cuda.amp.autocast if self.use_amp else nullcontext

        with context():

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

        if self.use_amp:
            self.amp_scaler.scale(l_total).backward()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """
        在 basicsr 中，所有 validation/test dataloader 的 batch size 都是 1，因此无法一张 png 里面存储多个图像
        分为两部分：
        - 计算指标
        - 保存图像
        """
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        detailed_metrics = pd.DataFrame()

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            lq_path = val_data['lq_path'][0]
            if lq_path.endswith('.taco'):
                img_name = osp.basename(lq_path.split(',')[0])
            else:
                img_name = osp.splitext(lq_path)[0]
            self.feed_data(val_data)
            self.test()

            # sr_image or gt_image HWC. Channel Order: RGB NIR. Value: 0-255
            visuals = self.get_current_visuals()            # cpu tensor B(RGBNIR)HW float32
            lq_img = tensor2ubyte_image(visuals['lq'])      # numpy H(BW)(RGBNIR) uint8
            sr_img = tensor2ubyte_image(visuals['result'])  # numpy H(BW)(RGBNIR) uint8
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2ubyte_image(visuals['gt'])
                metric_data['img2'] = gt_img
                del self.gt
            else:
                gt_img = None

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    result = calculate_metric(metric_data, opt_)
                    detailed_metrics.loc[img_name, name] = result
                    self.metric_results[name] += result

            if save_img:
                visual_fodler = self.opt['path']['visualization']
                rgb_path = osp.join(visual_fodler, "RGB", dataset_name, img_name)
                rgb_dict = {
                    "lq": lq_img[..., [2, 1, 0]],
                    "gt": gt_img[..., [2, 1, 0]] if gt_img is not None else None,
                    f"sr_{current_iter}": sr_img[..., [2, 1, 0]]
                }
                self.rswrite(rgb_path, rgb_dict)

                nir_path = osp.join(visual_fodler, "NIR", dataset_name, img_name)
                nir_dict = {
                    "lq": lq_img[..., [3]],
                    "gt": gt_img[..., [3]] if gt_img is not None else None,
                    f"sr_{current_iter}": sr_img[..., [3]]
                }
                self.rswrite(nir_path, nir_dict)

                if with_metrics:
                    save_csv_file = osp.join(visual_fodler, f"{dataset_name}_{current_iter}.csv")
                    detailed_metrics.to_csv(save_csv_file)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def rswrite(self, folder: str, filename_dict: dict):
        for key, value in filename_dict.items():
            save_file_path = osp.join(folder, f'{key}.png')
            if not osp.exists(save_file_path) and value is not None:
                imwrite(value, save_file_path)

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

