
from os import path as osp

from tqdm import tqdm

import pandas as pd
import torch
import torch._dynamo

from basicsr.metrics import calculate_metric
from basicsr.utils import zero_one_tensor_to_ubyte_numpy
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.srrs_model import SRRSModel


@MODEL_REGISTRY.register()
class TamingModel(SRRSModel):
    """
    Visualize pre-trained VQGAN model
    """

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.gt)
            self.net_g.train()


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
            lq_img = zero_one_tensor_to_ubyte_numpy(visuals['lq'])      # numpy H(BW)(RGBNIR) uint8
            sr_img = zero_one_tensor_to_ubyte_numpy(visuals['result'])  # numpy H(BW)(RGBNIR) uint8
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = zero_one_tensor_to_ubyte_numpy(visuals['gt'])
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
                    "lq": lq_img[..., :3],
                    "gt": gt_img[..., :3] if gt_img is not None else None,
                    f"sr_{current_iter}": sr_img[..., :3],
                }
                self.rswrite(rgb_path, rgb_dict, is_rgb_order=True)

                nir_path = osp.join(visual_fodler, "NIR", dataset_name, img_name)
                nir_dict = {
                    "lq": lq_img[..., [3]],
                    "gt": gt_img[..., [3]] if gt_img is not None else None,
                    f"sr_{current_iter}": sr_img[..., [3]],
                }
                self.rswrite(nir_path, nir_dict, is_rgb_order=False)

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