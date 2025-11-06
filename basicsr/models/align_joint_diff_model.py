import functools
from collections import OrderedDict
from contextlib import nullcontext

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
import torch._dynamo

from basicsr.archs import build_network
from basicsr.archs.alignae_unet_arch import AlignAutoencoderUNet
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.gaussian_diffusion_align import create_gaussian_diffusion
from basicsr.models.srrs_l2s_model import L2SSingleModel


@MODEL_REGISTRY.register()
class AlignJointDiffModel(L2SSingleModel):
    def __init__(self, opt):

        # init_training_settings 放在前面试试
        super().__init__(opt)
        self.net_g: AlignAutoencoderUNet

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

        self.align_loss = self.build_optional_loss(train_opt, 'encoder_opt')
        self.reconstruction_loss = self.build_optional_loss(train_opt, 'decoder_opt')
        self.sr_loss = self.build_optional_loss(train_opt, 'sr_opt')

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

        loss_dict = OrderedDict()

        context = torch.cuda.amp.autocast if self.use_amp else nullcontext

        with context():
            z_0 = self.base_diffusion.encode_first_stage(self.reg_input, self.net_g.net_ae(),
                                                         up_sample=False, no_grad=False)
            z_lr = self.base_diffusion.encode_first_stage(self.lq, self.net_g.net_lr(),
                                                         up_sample=True, no_grad=False)

            if self.opt["network_g"]["unet_args"]["cond_lq"]:
                model_kwargs = {'lq': z_lr}  #这里不 detach
            else:
                model_kwargs = None

            z0_pred = self.base_diffusion.forward_and_backward(self.net_g.net_u(),
                                                               hr=z_0.detach(),
                                                               lr=z_lr.detach(),
                                                               t=tt,
                                                               model_kwargs=model_kwargs,
                                                               noise=noise)

            rec = self.base_diffusion.decode_first_stage(z_0, self.net_g.net_ae(), no_grad=False)
            # 这里就不 detach 了，去噪网络直接瞄准像素空间结果。
            sr = self.base_diffusion.decode_first_stage(z0_pred, self.net_g.net_ae(), no_grad=False)

            # 1.alignment loss
            if self.align_loss is not None:
                loss_dict['align'] = self.align_loss(z_0, z_lr)

            # 2. diffusion loss
            if self.diffusion_loss is not None:
                loss_dict['diffusion'] = self.diffusion_loss(z0_pred, z_0.detach())

            # 3.reconstruction loss
            if self.reconstruction_loss is not None:
                loss_dict['reconstruction'] = self.reconstruction_loss(rec, self.gt)

            # 4. sr loss
            # diffusion 结果喂给 decoder, 增强训练 decoder.
            if self.sr_loss is not None:
                loss_dict['sr'] = self.sr_loss(sr, self.gt)

            # 记录 loss
            loss_dict = OrderedDict({key: value.mean() for key, value in loss_dict.items()})
            self.log_dict = self.reduce_loss_dict(loss_dict)

            loss = torch.stack(list(loss_dict.values())).sum()

        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            loss.backward()
            self.optimizer_g.step()



        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):

        def _get_output_and_diffused(model, lq):

            # 选择记录的时间步索引（最多记录4个阶段，确保包含最后一步）
            num_timesteps = self.base_diffusion.num_timesteps
            indices = np.linspace(
                0,
                num_timesteps,
                num_timesteps if num_timesteps < 5 else 4,
                endpoint=False,
                dtype=np.int64,
            ).tolist()

            if (num_timesteps - 1) not in indices:
                indices.append(num_timesteps - 1)

            # 初始化迭代计数器和结果容器
            num_iters = 0
            im_sr_all = None
            final_result = {}

            # 执行逐步采样过程
            for sample in self.base_diffusion.p_sample_loop_progressive(
                    y=lq,
                    model=model,
                    first_stage_model=None, # 在采样时，lq 依旧不需要 net_ae
                    noise=None,
                    clip_denoised=True, # 由于不需要 net_ae, 所以这里选 True
                    model_kwargs={'lq': lq},
                    device=self.device,
                    progress=False,
            ):

                if num_iters in indices:
                    # sample 的格式 {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}
                    result = {
                        'sample': self.base_diffusion.decode_first_stage(sample['sample'], self.net_g.net_ae()),
                        'latent_sample': sample['sample'],  # 原始潜空间张量
                    }
                    im_sr_progress = result['sample']
                    # 累积中间结果
                    if im_sr_all is None:
                        im_sr_all = im_sr_progress
                    else:
                        im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    # 更新最终输出
                    final_result = result

                num_iters += 1

            # 重排维度以便后续处理
            im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=lq.shape[1])

            return final_result['sample'], final_result['latent_sample'], im_sr_all

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.z0 = self.base_diffusion.encode_first_stage(self.reg_input, self.net_g_ema.net_ae(),
                                                                 up_sample=False)
                self.z0_output = self.base_diffusion.decode_first_stage(self.z0, self.net_g_ema.net_ae())
                self.output, self.z0_pred, self.sr_all = _get_output_and_diffused(self.net_g_ema.net_u(), self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.z0 = self.base_diffusion.encode_first_stage(self.reg_input, self.net_g.net_ae(),
                                                                 up_sample=False)
                self.z0_output = self.base_diffusion.decode_first_stage(self.z0, self.net_g.net_ae())
                self.output, self.z0_pred, self.sr_all = _get_output_and_diffused(self.net_g.net_u(), self.lq)
            self.net_g.train()

    def get_current_visuals(self, current_iter: int):
        out_dict = super().get_current_visuals(current_iter)
        out_dict[f"z0_{current_iter}"] = self.z0.detach().cpu()
        out_dict[f"sr_z0_{current_iter}"] = self.z0_output.detach().cpu()

        out_dict[f"z0_pred_{current_iter}"] = self.z0_pred.detach().cpu()
        out_dict[f"sr_all_{current_iter}"] = self.sr_all.detach().cpu()
        return out_dict
