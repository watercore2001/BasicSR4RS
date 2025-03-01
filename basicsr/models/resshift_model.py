import functools
from collections import OrderedDict
from contextlib import nullcontext
from os import path as osp

from tqdm import tqdm
from einops import rearrange
import numpy as np
import pandas as pd
import torch
import torch._dynamo

from basicsr.metrics import calculate_metric
from basicsr.archs import build_network
from basicsr.utils import get_root_logger, tensor2ubyte_image
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.gaussian_diffusion import create_gaussian_diffusion
from .sr_rs_model import SRRSModel


@MODEL_REGISTRY.register()
class ResShiftModel(SRRSModel):
    def __init__(self, opt):

        # build net_g (swin unet)
        super().__init__(opt)

        # build autoencoder
        if "autoencoder" in self.opt:
            # define autoencoder
            self.autoencoder = build_network(opt["autoencoder"])
            self.autoencoder = self.model_to_device(self.autoencoder)

            # load weight pth
            load_path_ae = self.opt['path'].get('pretrain_network_ae', None)
            if load_path_ae is not None:
                self.load_network(self.autoencoder, load_path_ae, True, None)

            # eval autoencoder
            for params in self.autoencoder.parameters():
                params.requires_grad_(False)
            self.autoencoder.eval()

            self.autoencoder_attr_accessor = self.get_bare_model(self.autoencoder)


        # build diffusion
        diffusion_params = opt["diffusion"]
        self.base_diffusion = create_gaussian_diffusion(**diffusion_params)

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

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_record_step_loss(self, losses, tt):
        num_timesteps = self.base_diffusion.num_timesteps
        record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]  # [1, 8, 15] for 15

        # 如果有两个 loss, step为 15, 那么形状为 {2: 3} 的字典
        loss_mean = {key: torch.zeros(size=(len(record_steps),), dtype=torch.float64, device=self.device)
                     for key in losses.keys()}
        # 记录每个step在batch中所对应的item数量，用来平均每个step的loss
        loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64, device=self.device)

        # 记录数据
        for jj in range(len(record_steps)):
            for key, value in losses.items():
                index = record_steps[jj] - 1
                mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                current_loss = torch.sum(value.detach() * mask)
                loss_mean[key][jj] += current_loss.item()

            loss_count[jj] += mask.sum().item()

        # 取平均
        if torch.any(loss_count == 0):
            loss_count += 1e-4

        for key in losses.keys():
            loss_mean[key] /= loss_count

        # 最终log显示的损失
        loss_dict = OrderedDict()

        for jj, current_record in enumerate(record_steps):
            for key, values in loss_mean.items():
                final_key = key + str(current_record)  # eg: mse for step8
                final_value = values[jj]
                loss_dict[final_key] = final_value

        return loss_dict

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # random timesteps for training denoising
        batch_size = self.gt.shape[0]
        tt = torch.randint(
            0, self.base_diffusion.num_timesteps,
            size=(batch_size,),
            device=self.device,
        )

        # random noise for added to gt
        latent_downsamping_factor = 2 ** (len(self.opt["autoencoder"]["ddconfig"]["ch_mult"]) - 1)
        latent_resolution = self.gt.shape[-1] // latent_downsamping_factor
        if "autoencoder" in self.opt:
            noise_chn = self.opt["autoencoder"]["embed_dim"]
        else:
            noise_chn = self.gt.shape[1]
        noise = torch.randn(
            size=(batch_size, noise_chn,) + (latent_resolution,) * 2,
            device=self.device,
        )

        if self.opt["network_g"]["cond_lq"]:
            model_kwargs = {'lq': self.lq}
        else:
            model_kwargs = None

        loss_computer = functools.partial(
            self.base_diffusion.training_losses,
            self.net_g,
            self.gt,
            self.lq,
            tt,
            first_stage_model=self.autoencoder_attr_accessor,
            model_kwargs=model_kwargs,
            noise=noise,
        )

        context = torch.cuda.amp.autocast if self.use_amp else nullcontext

        with context():
            loss_dict_each_item, z_t, z0_pred = loss_computer()
            loss_dict_each_item['loss'] = loss_dict_each_item['mse']
            loss = loss_dict_each_item['loss'].mean()

        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            loss.backward()
            self.optimizer_g.step()

        # loss_dict = self.get_record_step_loss(loss_dict_each_item, tt)
        # 就用简单的损失算了
        loss_dict = OrderedDict({key: value.mean() for key, value in loss_dict_each_item.items()})
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):

        def _get_output_and_diffused(model, lq):

            # 记录的中间步骤
            indices = np.linspace(
                0,
                self.base_diffusion.num_timesteps,
                self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                endpoint=False,
                dtype=np.int64,
            ).tolist()
            if not (self.base_diffusion.num_timesteps - 1) in indices:
                indices.append(self.base_diffusion.num_timesteps - 1)

            # 开始迭代去噪
            num_iters = 0

            for sample in self.base_diffusion.p_sample_loop_progressive(
                    y=lq,
                    model=model,
                    first_stage_model=self.autoencoder_attr_accessor,
                    noise=None,
                    clip_denoised=True if self.autoencoder_attr_accessor is None else False,
                    model_kwargs={'lq': lq},
                    device=self.device,
                    progress=False,
            ):
                # sample 的格式 {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}
                sample_decode = {}
                if num_iters in indices:
                    for key, value in sample.items():
                        if key in ['sample', ]:
                            sample_decode[key] = self.base_diffusion.decode_first_stage(
                                value,
                                self.autoencoder_attr_accessor,)
                    im_sr_progress = sample_decode['sample']
                    if num_iters + 1 == 1:
                        im_sr_all = im_sr_progress
                    else:
                        im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                num_iters += 1

            im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=lq.shape[1])

            return sample_decode['sample'], im_sr_all

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, self.sr_all = _get_output_and_diffused(self.net_g_ema, self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, self.sr_all = _get_output_and_diffused(self.net_g, self.lq)
            self.net_g.train()

    def get_current_visuals(self):
        out_dict = super().get_current_visuals()
        out_dict["sr_all"] = self.sr_all.detach().cpu()
        return out_dict

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
            lq_img = tensor2ubyte_image(visuals['lq'], )      # numpy H(BW)(RGBNIR) uint8
            sr_all = tensor2ubyte_image(visuals['sr_all'])  # numpy H(BW)(RGBNIR) uint8
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
                    f"sr_{current_iter}": sr_img[..., [2, 1, 0]],
                    f"all_{current_iter}": sr_all[..., [2, 1, 0]],
                }
                self.rswrite(rgb_path, rgb_dict)

                nir_path = osp.join(visual_fodler, "NIR", dataset_name, img_name)
                nir_dict = {
                    "lq": lq_img[..., [3]],
                    "gt": gt_img[..., [3]] if gt_img is not None else None,
                    f"sr_{current_iter}": sr_img[..., [3]],
                    f"all_{current_iter}": sr_all[..., [3]],
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