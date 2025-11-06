import functools
from collections import OrderedDict
from contextlib import nullcontext

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
import torch._dynamo

from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.gaussian_diffusion_registration import create_gaussian_diffusion
from basicsr.models.srrs_l2s_model import L2SSingleModel


@MODEL_REGISTRY.register()
class AlignFrozenDiffModel(L2SSingleModel):
    def __init__(self, opt):

        # init_training_settings 放在前面试试
        super().__init__(opt)

        # 推理和训练都需要的属性，不能放进 init_training_settings
        # build autoencoder
        if "autoencoder" in self.opt:
            # define autoencoder
            self.autoencoder = build_network(opt["autoencoder"])
            self.autoencoder = self.model_to_device(self.autoencoder)

            # load weight pth
            load_path_ae = self.opt['path'].get('pretrain_network_ae', None)
            if load_path_ae is not None:
                self.load_network(self.autoencoder, load_path_ae, True, "params_ema")

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

    def get_record_step_loss(self, losses, tt):
        num_timesteps = self.base_diffusion.num_timesteps
        record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]  # [1, 8, 15] for 15

        # 如果有两个 loss, step 为 15, 那么形状为 {2: 15} 的字典
        loss_mean = {key: torch.zeros(size=(len(record_steps),), dtype=torch.float64, device=self.device)
                     for key in losses.keys()}
        # 记录每个 step 在 batch 中所对应的 item 数量，用来平均每个 step 的 loss
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

        # Generate Gaussian noise with spatial dimensions matching ground truth
        noise = torch.randn(
            self.gt.shape,
            device=self.device
        )

        if self.opt["network_g"]["cond_lq"]:
            model_kwargs = {'lq': self.lq}
        else:
            model_kwargs = None

        loss_computer = functools.partial(
            self.base_diffusion.training_losses,
            self.net_g,
            self.reg_input,
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
                    first_stage_model=None,
                    noise=None,
                    clip_denoised=True,
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
