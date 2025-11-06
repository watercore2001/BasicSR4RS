import pyiqa
import numpy as np
import torch
from basicsr.utils.registry import METRIC_REGISTRY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建 LPIPS 评估器
iqa_metric = pyiqa.create_metric('lpips', device=DEVICE)

# from torchvision.transforms.functional import normalize
# from basicsr.utils import img2tensor
# import lpips
# from basicsr.metrics.metric_util import reorder_image, to_y_channel
# @METRIC_REGISTRY.register()
# def calculate_rs_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False,
#                        input_bands: list[int] = (2, 1, 0), **kwargs):
#
#     """Calculate LPIPS.
#     Ref: https://github.com/xinntao/BasicSR/pull/367
#     Args:
#         img (ndarray): Images with range [0, 255]. BGR
#         img2 (ndarray): Images with range [0, 255]. BGR
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the PSNR calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
#         input_bands: BGR bands index in img and img2
#     Returns:
#         float: LPIPS result.
#     """
#     assert img.shape == img2.shape, (f'Image shapes are differnet: {img.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
#     img = reorder_image(img, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)
#
#     img = img[..., input_bands]
#     img2 = img2[..., input_bands]
#
#     if crop_border != 0:
#         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
#
#     if test_y_channel:
#         img = to_y_channel(img)
#         img2 = to_y_channel(img2)
#
#     # start calculating LPIPS metrics
#
#     mean = [0.5, 0.5, 0.5]
#     std = [0.5, 0.5, 0.5]
#
#     img_gt = img2 / 255.
#     img_restored = img / 255.
#
#     img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
#     # norm to [-1, 1]
#     normalize(img_gt, mean, std, inplace=True)
#     normalize(img_restored, mean, std, inplace=True)
#
#     # calculate lpips
#     img_gt = img_gt.to(DEVICE)
#     img_restored = img_restored.to(DEVICE)
#     loss_alex = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)  # RGB, normalized to [-1,1]
#     loss_alex.eval()
#     lpips_val = loss_alex(img_restored.unsqueeze(0), img_gt.unsqueeze(0))
#
#     return lpips_val.detach().cpu().numpy().mean()


def compute_lpips_score(img: np.ndarray, img2: np.ndarray, **kwargs) -> float:
    """
    计算两个图像的 LPIPS 评分。

    参数:
        img (np.ndarray): 第一个图像，形状为 (H, W)
        img2 (np.ndarray): 第二个图像，形状为 (H, W)
        device (torch.device): 要放置张量的设备（如 'cuda' 或 'cpu'）

    返回:
        float: LPIPS 评分
    """
    # 检查输入形状
    assert img.ndim == 2 and img2.ndim == 2, "输入图像必须是 (H, W) 形状的灰度图"

    # 转换为 float32 并归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # 扩展为 RGB 通道
    img_rgb = np.stack([img] * 3, axis=0)  # (3, H, W)
    img2_rgb = np.stack([img2] * 3, axis=0)

    # 添加 batch 维度并转换为张量
    img_tensor = torch.from_numpy(img_rgb).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
    img2_tensor = torch.from_numpy(img2_rgb).unsqueeze(0).to(DEVICE)

    # 计算分数并返回 float
    score = iqa_metric(img_tensor, img2_tensor)
    return float(score.item())


@METRIC_REGISTRY.register()
def calculate_lpips_band(img, img2, crop_border, band, input_order='HWC', **kwargs):
    """Calculate LPIPS for a specific band (channel) of the input images.

    Args:
        img (ndarray): Input image with range [0, 255].
        img2 (ndarray): Reference image with range [0, 255].
        crop_border (int): Number of pixels to crop from each border.
        band (int): Index of the band (channel) to calculate LPIPS on.
        input_order (str): 'HWC' or 'CHW'. Default: 'HWC'.
        **kwargs: Additional arguments passed to compute_lpips_score.

    Returns:
        float: LPIPS score for the specified band.
    """
    assert img.shape == img2.shape, f'Image shapes are different: {img.shape}, {img2.shape}.'

    # Extract band
    if input_order == 'HWC':
        assert band < img.shape[2], f'Band index {band} out of range for shape {img.shape}.'
        img_band = img[:, :, band]
        img2_band = img2[:, :, band]
    elif input_order == 'CHW':
        assert band < img.shape[0], f'Band index {band} out of range for shape {img.shape}.'
        img_band = img[band, :, :]
        img2_band = img2[band, :, :]
    else:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW".')

    # Crop borders
    if crop_border > 0:
        if input_order == 'HWC':
            img_band = img_band[crop_border:-crop_border, crop_border:-crop_border]
            img2_band = img2_band[crop_border:-crop_border, crop_border:-crop_border]
        else:  # CHW
            img_band = img_band[:, crop_border:-crop_border, crop_border:-crop_border]
            img2_band = img2_band[:, crop_border:-crop_border, crop_border:-crop_border]

    # Compute LPIPS score
    return compute_lpips_score(img_band, img2_band, **kwargs)


@METRIC_REGISTRY.register()
def calculate_lpips_none(**kwargs):
    return -1

if __name__ == '__main__':
    import numpy as np

    # 假设你有两张图像 img1 和 img2，形状为 (H, W, C)，值范围 [0, 255]
    img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # 设置参数
    crop_border = 4  # 裁剪边缘像素
    band_index = 1  # 比较第2个通道（如 G 通道）
    input_order = 'HWC'  # 图像格式为 HWC

    # 调用函数
    lpips_score = calculate_lpips_band(
        img1, img2,
        crop_border=crop_border,
        band=band_index,
        input_order=input_order,
    )

    print(f"LPIPS score for band {band_index}: {lpips_score:.4f}")