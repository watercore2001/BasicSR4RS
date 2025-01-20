import torch
import numpy as np
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, save_lq_sr_image, save_all_image
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SRRSModel(SRModel):
    """Base SR model for single rs image super-resolution."""

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
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
            visuals = self.get_current_visuals()
            lq_img = tensor2img([visuals['lq']], rgb2bgr=False)
            sr_img = tensor2img([visuals['result']], rgb2bgr=False)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=False)
                metric_data['img2'] = gt_img
                del self.gt
            else:
                gt_img = None

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                visual_fodler = self.opt['path']['visualization']
                rgb_path = osp.join(visual_fodler, "RGB")
                self.rswrite(lq_img[...,[2,1,0]], sr_img[...,[2,1,0]], gt_img[...,[2,1,0]],
                             current_iter, rgb_path, img_name, dataset_name)

                nir_path = osp.join(visual_fodler, "NIR")
                self.rswrite(lq_img[..., [3]], sr_img[..., [3]], gt_img[..., [3]],
                             current_iter, nir_path, img_name, dataset_name)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
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


    def rswrite(self, lq_img: np.ndarray, sr_img: np.ndarray, gt_img: np.ndarray, current_iter: int, folder: str,
                img_name: str, dataset_name: str):
        """
        lq_img: uin8 [0-255], HWC, Channels: BGR or single
        sr_img: uin8 [0-255], HWC, Channels: BGR or single
        gt_img: uin8 [0-255], HWC, Channels: BGR or single
        current_iter: int
        folder: str
        img_name: str
        dataset_name: str
        """

        if self.opt['is_train']:
            lq_path = osp.join(folder, img_name, "lq.png")
            gt_path = osp.join(folder, img_name, "gt.png")
            sr_path = osp.join(folder, img_name, f'sr_{current_iter}.png')
        else:
            lq_path = osp.join(folder, dataset_name, img_name, "lq.png")
            gt_path = osp.join(folder, dataset_name, img_name, "gt.png")

            if self.opt['val']['suffix']:
                # add suffix is good for comparing with other model
                suffix = self.opt['val']['suffix']
            else:
                suffix = self.opt["name"]

            sr_path = osp.join(folder, dataset_name, img_name, f'sr_{suffix}.png')

            # save all image in one picture
            all_path = osp.join(folder, dataset_name, f'{img_name}_{suffix}.png')
            if gt_img is None:
                save_lq_sr_image(lq_img, sr_img, all_path)
            else:
                save_all_image(lq_img, sr_img, gt_img, all_path)

        if not osp.exists(lq_path):
            imwrite(lq_img, lq_path)

        if gt_img is not None and not osp.exists(gt_path):
            imwrite(gt_img, gt_path)

        imwrite(sr_img, sr_path)



