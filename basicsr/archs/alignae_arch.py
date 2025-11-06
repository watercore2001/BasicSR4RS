import torch.nn as nn
from basicsr.archs.arch_util import make_layer, CAB
from basicsr.utils.registry import ARCH_REGISTRY


class AlignNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=4):
        """
        Args:
            num_in_ch (int): 输入图像通道数
            num_out_ch (int): 输出图像通道数
            num_feat (int): 中间特征通道数
            num_block (int): 残差块数量（网络深度）
        """
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(CAB, num_block, num_feat=num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 1, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        x = self.body(x)
        x = self.conv_last(x)
        return x


@ARCH_REGISTRY.register()
class LREncoder(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=8):
        super().__init__()
        self.lr_encoder = AlignNet(num_in_ch, num_out_ch, num_feat, num_block)

    def encode(self, x):
        return self.lr_encoder(x)


@ARCH_REGISTRY.register()
class AlignAutoencoder(nn.Module):
    def __init__(self, num_in_ch: int, num_out_ch: int, num_feat=64, num_block=8):
        super().__init__()
        self.align_encoder = AlignNet(num_in_ch, num_out_ch, num_feat, num_block)
        self.decoder = AlignNet(num_out_ch, num_out_ch, num_feat, num_block)

    def encode(self, x):
        return self.align_encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        stage1_sr = self.encode(x)
        stage2_sr = self.decode(stage1_sr)
        return {"stage1": stage1_sr, "stage2": stage2_sr}
