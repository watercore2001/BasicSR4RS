import pyiqa
import torch
from PIL import Image
import numpy as np

from basicsr.metrics.lpips import calculate_lpips_band


def tes1():
    # list all available metrics
    print(pyiqa.list_models())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create metric with default setting
    iqa_metric = pyiqa.create_metric('lpips', device=device)

    # check if lower better or higher better
    print(iqa_metric.lower_better)

    # example for iqa score inference
    # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
    N, H, W = 4, 128, 128
    img_tensor_1 = torch.rand((N, 3, H, W), dtype=torch.float32)
    img_tensor_2 = torch.rand((N, 3, H, W), dtype=torch.float32)
    score_fr = iqa_metric(img_tensor_1, img_tensor_2)
    print(score_fr)


def two_image(img_path1, img_path2):
    # 使用 PIL 读取图像并转换为 RGB
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')

    # 转换为 NumPy 数组，形状为 (H, W, C)，值范围 [0, 255]
    img1 = np.array(img1)
    img2 = np.array(img2)

    # 📊 计算三个波段的 LPIPS 分数
    scores = []
    for band in range(3):  # R, G, B
        score = calculate_lpips_band(img1, img2, band=band, crop_border=4, input_order='HWC')
        scores.append(score)

    # 📈 求平均值
    average_score = sum(scores) / len(scores)
    print(f"LPIPS scores per band: {scores}")
    print(f"Average LPIPS score: {average_score:.4f}")

img_path1 = "/mnt/code/deep_learning/BasicSR/tests/data/imgs/ex_p0.png"
img_path2 = "/mnt/code/deep_learning/BasicSR/tests/data/imgs/ex_p1.png"
img_path3 = "/mnt/code/deep_learning/BasicSR/tests/data/imgs/ex_ref.png"

two_image(img_path2, img_path3)

