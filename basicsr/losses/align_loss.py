from typing import Tuple, Union, Optional, Literal
from torch.nn import functional as F
import torch
from torch import Tensor, nn
import numpy as np

from basicsr.utils.registry import LOSS_REGISTRY

def lanczos_kernel(
    dx : Union[float, Tensor],
    a : int = 3,
    N : int = None,
    dtype : Optional[type] = None,
    device : Optional[torch.device] = None
) -> Tensor:
    '''
    Generates 1D Lanczos kernels for translation and interpolation.

    Args:
        dx : float, tensor (batch_size, 1), the translation in pixels to shift an image.
        a : int, number of lobes in the kernel support.
            If N is None, then the width is the kernel support (length of all lobes),
            S = 2(a + ceil(dx)) + 1.
        N : int, width of the kernel.
            If smaller than S then N is set to S.

    Returns:
        k: tensor (?, ?), lanczos kernel
    '''

    if not torch.is_tensor(dx):
        dx = torch.tensor(dx, dtype=dtype, device=device)

    if device is None:
        device = dx.device

    if dtype is None:
        dtype = dx.dtype

    D = dx.abs().ceil().int()
    S = 2 * (a + D) + 1  # width of kernel support

    S_max = S.max() if hasattr(S, 'shape') else S

    if (N is None) or (N < S_max):
        N = S

    Z = (N - S) // 2  # width of zeros beyond kernel support

    start = (-(a + D + Z)).min()
    end = (a + D + Z + 1).max()
    x = torch.arange(start, end, dtype=dtype, device=device).view(1, -1) - dx
    px = (np.pi * x) + 1e-3

    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / a)

    k = a * sin_px * sin_pxa / px**2  # sinc(x) masked by sinc(x/a)

    return k


class ShiftConv2d(nn.Module):
    '''
    A Conv2d layer and generates are shifted versions of x, with shifts
    from `start` to `end` with stepsize `step` on the last two dimensions.
    '''

    def __init__(self, start: float, end: float, step: float) -> None:
        """
        Args:
            start (float): relative start shift (in pixels).
            end (float): relative end shift (in pixels).
            step (float): (sub-)pixel shift of each step from start to end.
        """

        super().__init__()

        self.start = float(start)
        self.end = float(end)
        self.step = float(step)

#         if (step == 1) and (start == -end) and ((end - start) % 1 == 0):
#             K_y, K_x = self.separable_shift_kernels(w=w)
#             K_y, K_x = K_y[:, None, None], K_x[:, None, None]
#         else:
        K_y, K_x = self.separable_lanczos_kernels(self.start, self.end, self.step)

        o, _, _, h, _ = K_y.shape
        self.conv2d_yshift = nn.Conv3d(in_channels=1, out_channels=o, kernel_size=(1, h, 1),
                                    padding=(0, h//2, 0), bias=False, padding_mode='zeros')
        self.conv2d_yshift.weight.data = K_y  # Fix and freeze the shift-kernel
        self.conv2d_yshift.requires_grad_(False)
        self.register_buffer("K_y", K_y)

        o, _, _, _, w = K_x.shape
        self.conv2d_xshift = nn.Conv3d(in_channels=1, out_channels=o, kernel_size=(1, 1, w),
                                    padding=(0, 0, w//2), bias=False, padding_mode='zeros')
        self.conv2d_xshift.weight.data = K_x
        self.conv2d_xshift.requires_grad_(False)
        self.register_buffer("K_x", K_x)


#     ## This functionality is now covered by `separable_lanczos_kernels` and hence it is deprecated.
#     def separable_shift_kernels(self, w : int) -> Tuple[Tensor, Tensor]:
#         '''
#         Makes 2*w 1D convolution kernels (w, 1) for discrete shifts.

#         Args:
#             w (int): shift in number of pixels.

#         Returns:
#             A 2-tuple of tensors ((w, 1), (1, w))
#         '''

#         K_y = torch.zeros(w, w, 1)  # (num_kernels, H, W)
#         K_x = torch.zeros(w, 1, w)
#         for i in range(w):
#             K_y[i, i, 0] = 1
#             K_x[i, 0, i] = 1

#         return K_y, K_x


    def separable_lanczos_kernels(
        self,
        start : float,
        end : float,
        step : float,
    ) -> Tuple[Tensor, Tensor]:
        '''
        Makes two sets of 1D convolution kernels for y- and x-axis shifts.

        Args:
            start (float): relative start shift (in pixels).
            end (float): relative end shift (in pixels).
            step (float): (sub-)pixel shift of each step from start to end.

        Returns:
            A 2-tuple of tensors ((k, 1), (1, k))
        '''

        shift = torch.arange(start, end + 1e-3, step)[:, None]
        K_ = lanczos_kernel(shift, a=3)
        K_y = K_[:, None, None, :, None]
        K_x = K_[:, None, None, None, :]

        return K_y, K_x

    def forward(self, x : Tensor) -> Tensor:
        x = x[:, None]
        xs = self.conv2d_yshift(x)
        B, S, C, H, W = xs.shape
        xs = xs.view(B * S, 1, C, H, W)
        xs = self.conv2d_xshift(xs)
        _, _, _, H, W = xs.shape
        xs = xs.view(B, -1, C, H, W)
        return xs


@LOSS_REGISTRY.register()
class RegisteredLoss(nn.Module):
    '''
    Applies a loss func to shifted versions of y and forwards the min of the shifted losses.
    Initial version: https://gitlab.com/frontierdevelopmentlab/fdl-us-2020-droughts/xstream/-/blob/registered-loss/ml/src/loss.py#L73-130
    '''

    def __init__(
        self,
        start: float,
        end: float,
        step: float,
        loss_func: Literal["l1", "l2", "mse"],
        loss_weight: float = 1.0,
        reduction='mean',
    ) -> None:
        """
        Args:
            start (float): relative start shift (in pixels).
            end (float): relative end shift (in pixels).
            step (float): (sub-)pixel shift of each step from start to end.
            loss_func (callable): loss function to apply at each pixel of each channel.
                Hint: use the `reduction='none'` option if the loss supports it.
            reduction (str, default='mean'): Reduction to apply along the batch dimension.
                One of ['mean'|'sum'|'none']. 'none' applies no reduction.
            **loss_kwargs (dict): arguments passed to `loss_func`.
        """

        super().__init__()

        self._shiftconv2d = ShiftConv2d(start, end, step)
        self.start = float(start)
        self.end = float(end)
        self.step = float(step)

        if loss_func.lower() == 'l1':
            self.loss_func = nn.L1Loss(reduction='none')
        elif loss_func.lower() in ['mse', 'l2']:
            self.loss_func = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss_func: {loss_func}. Choose from ['l1', 'mse']")

        self.loss_weight = loss_weight
        self.reduction = reduction

    def _shifted_loss(self, y_pred: Tensor, y : Tensor) -> torch.float32:
        '''
        Shifted versions of the loss.

        Args:
            y_pred : torch.Tensor (B, C, H, W).
            y : torch.Tensor (B, C, H, W).

        Returns:
            torch.Tensor (B, num_shifts)
        '''

        wy = self._shiftconv2d.conv2d_yshift.weight.shape[-2] // 2
        wx = self._shiftconv2d.conv2d_xshift.weight.shape[-1] // 2

        ## Shifted versions of y: (B, num_shifts, C, H, W).
        y_pred_shifted = self._shiftconv2d(y_pred)
        ## Do not evaluate loss at the border strips.
        y_pred_shifted = y_pred_shifted[..., wy:-wy, wx:-wx]

        ## Broadcastable view for loss_func. expand_as creates a view (no copying).
        _y = y[:, None, :, wy:-wy, wx:-wx]
        _y = _y.expand_as(y_pred_shifted)

        ## Element-wise loss.
        loss = self.loss_func(y_pred_shifted, _y)

        return loss.mean(dim=(-3, -2, -1))  # Reduce along C, H, W dims.

    def registered_loss(self, y_pred: Tensor, y: Tensor) -> torch.float32:
        '''
        Version of the loss where only the min of each input in the mini-batch is forwarded.
        '''
        B = y.shape[0]  # Batch size
        loss_all_shifts = self._shifted_loss(y_pred, y)
        i_min_loss = loss_all_shifts.argmin(dim=1)  # Argmin loss
        min_loss = loss_all_shifts[range(B), i_min_loss]

        if self.reduction == 'mean':
            loss = min_loss.mean()
        elif self.reduction == 'sum':
            loss = min_loss.sum()
        elif self.reduction == 'none':
            loss = min_loss
        else:
            raise NotImplementedError(f"Expected `reduction` values: ['mean'|'sum'|'none']. Got {self.reduction}.")

        return self.loss_weight * loss

    def forward(self, y_pred: Tensor, y: Tensor) -> torch.float32:
        return self.registered_loss(y_pred, y)


@LOSS_REGISTRY.register()
class EncoderLoss(nn.Module):
    """Encoder loss with strategy control.

    Args:
        loss_weight (float): Overall loss weight. Default: 1.0.
        strategy (str): Loss computation strategy.
        reduction (str): Reduction mode: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, strategy='gt', reduction='mean'):
        super().__init__()
        if strategy not in ['gt', "lq"]:
            raise ValueError(f"Unsupported loss strategy {strategy}")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}')

        self.loss_weight = loss_weight
        self.strategy = strategy
        self.reduction = reduction

    def forward(self, z_start, gt, lq):
        """
        Args:
            z_start (Tensor): Encoder output.
            gt (Tensor): Ground truth tensor.
            lq (Tensor, optional): Low-quality input tensor.
        Returns:
            Tensor: Computed loss based on strategy.
        """
        if self.strategy == 'gt':
            loss = F.mse_loss(z_start, gt, reduction=self.reduction)

        elif self.strategy == 'lq':
            # 上采样 lq 到与 z_start 相同的空间尺寸
            lq_upsampled = F.interpolate(lq, size=z_start.shape[2:], mode='bilinear', align_corners=False)
            loss = F.mse_loss(z_start, lq_upsampled, reduction=self.reduction)

        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        return self.loss_weight * loss
