from torch import nn
import torch.nn.functional as functional
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SRCNN(nn.Module):
    def __init__(self, num_in_ch: int, num_out_ch: int, upscale: int):
        super(SRCNN, self).__init__()
        self.upscale = upscale
        self.conv1 = nn.Conv2d(num_in_ch, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_out_ch, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = functional.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=True)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
