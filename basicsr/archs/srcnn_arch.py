import torch
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


# @ARCH_REGISTRY.register()
# class SRCNNL2S(nn.Module):
#     num_in_ch = 6
#     num_out_ch = 6
#
#     def __init__(self):
#         super(SRCNNL2S, self).__init__()
#         self.rgb_upscale = 3
#         self.nss_upscale = 2
#
#         self.conv1 = nn.Conv2d(self.num_in_ch, 64, kernel_size=9, padding=9 // 2)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
#         self.conv3 = nn.Conv2d(32, self.num_out_ch, kernel_size=5, padding=5 // 2)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.rgb_head = nn.Sequential(
#             nn.Conv2d(self.num_out_ch, 3 * 9, kernel_size=3, padding=1),
#             nn.PixelShuffle(upscale_factor=3)
#         )
#
#         self.nss_head = nn.Sequential(
#             nn.Conv2d(self.num_out_ch, 3 * 4, kernel_size=3, padding=1),
#             nn.PixelShuffle(upscale_factor=3)
#         )
#
#     def forward(self, x_rgb, x_nss):
#         x = torch.cat([x_rgb, x_nss], dim=1)
#
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
#
#         rgb_out = self.rgb_head(x)
#         nss_out = self.nss_head(x)
#
#         return rgb_out, nss_out
