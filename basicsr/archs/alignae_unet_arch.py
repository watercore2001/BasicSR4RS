import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.alignae_arch import AlignAutoencoder, LREncoder
from basicsr.archs.unet_arch import UNetModelSwin


class AlignAutoencoderUNet(nn.Module):
    unet_class = None
    align_autoencoder_class = None
    lr_encoder_class = None

    def __init__(self, unet_args: dict, align_autoencoder_args: dict, lr_encoder_args: dict) -> None:
        super(AlignAutoencoderUNet, self).__init__()

        assert self.unet_class is not None, "unet_class must be defined in subclass"
        self.unet = self.unet_class(**unet_args)

        assert self.align_autoencoder_class is not None, "align_autoencoder_class must be defined in subclass"
        self.align_autoencoder = self.align_autoencoder_class(**align_autoencoder_args)

        assert self.lr_encoder_class is not None, "lr_encoder_class must be defined in subclass"
        self.lr_encoder = self.lr_encoder_class(**lr_encoder_args)

    def net_lr(self):
        return self.lr_encoder

    def net_ae(self):
        return self.align_autoencoder

    def net_u(self):
        return self.unet

    def forward(self, x):
        # 应该间接调用其他网络
        raise NotImplementedError


@ARCH_REGISTRY.register()
class ResNetAE_SwinUNet(AlignAutoencoderUNet):
    unet_class = UNetModelSwin
    align_autoencoder_class = AlignAutoencoder
    lr_encoder_class = LREncoder
