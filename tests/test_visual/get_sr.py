import numpy as np
import cv2
import os

def shift_and_blur(
    image: np.ndarray,
    shift_x: int = 1,
    shift_y: int = 1,
    blur_kernel: int = 3,
    blur_sigma: float = 0.5,
    save_path: str = "output.png"
) -> np.ndarray:
    """
    将图像偏移指定像素并加上高斯模糊，并保存结果。

    参数:
        image (np.ndarray): 输入图像（灰度或RGB）
        shift_x (int): 水平方向偏移像素数
        shift_y (int): 垂直方向偏移像素数
        blur_kernel (int): 高斯模糊核大小（必须为奇数）
        blur_sigma (float): 高斯模糊标准差
        save_path (str): 保存路径（支持相对或绝对路径）

    返回:
        np.ndarray: 处理后的图像
    """
    # 1. 仿射偏移
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # 2. 高斯模糊
    if blur_kernel % 2 == 0:
        blur_kernel += 1  # 保证为奇数
    blurred = cv2.GaussianBlur(shifted, (blur_kernel, blur_kernel), sigmaX=blur_sigma)

    # 3. 保存图像
    ext = os.path.splitext(save_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        cv2.imwrite(save_path, cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return blurred


import matplotlib.pyplot as plt

img = cv2.imread('./crop/hr.png')  # 读取图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB

result = shift_and_blur(
    img,
    shift_x=1.5,
    shift_y=1.5,
    blur_kernel=5,
    blur_sigma=0.5,
    save_path="./crop/sr.png"
)
