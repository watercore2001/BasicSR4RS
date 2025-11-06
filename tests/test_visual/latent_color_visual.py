import cv2
import numpy as np
import os


def crop_top_left_quarter(input_folder, output_folder="crop"):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图片扩展名
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in valid_exts:
            # 读取图像
            img = cv2.imread(file_path)
            if img is None:
                print(f"跳过无法读取的文件: {filename}")
                continue

            h, w = img.shape[:2]
            cropped = img[0:h//4, 0:w//4]  # 左上角四分之一

            # 保存裁剪后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped)
            print(f"已保存: {output_path}")

crop_top_left_quarter("origin")

def generate_color_tones(input_path, output_dir, saturation_scale=1.3, brightness_scale=1.1):
    # 常见色调及对应 Hue 值（OpenCV 范围）
    tone_hues = {
        "red": 0,
        "orange": 15,
        "yellow": 30,
        "yellow_green": 45,
        "green": 70,
        "cyan": 95,
        "blue": 120,
        "indigo": 135,
        "purple": 150,
        "magenta": 165
    }

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")

    # 转换到 HSV 色彩空间
    hsv_original = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for tone_name, hue_value in tone_hues.items():
        hsv = hsv_original.copy()

        # 设置 Hue 通道为目标色调
        hsv[:, :, 0] = hue_value

        # 增强饱和度和亮度
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_scale, 0, 255)

        # 转回 BGR 色彩空间
        toned_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 保存图像
        output_path = os.path.join(output_dir, f"{tone_name}.png")
        cv2.imwrite(output_path, toned_img)
        results[tone_name] = toned_img

    return results


# generate_color_tones("crop/lr.png", "color_lr")
# generate_color_tones("crop/hr.png", "color_hr")


def add_noise_and_save_maps(input_path, output_dir, noise_levels=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")

    img_float = img.astype(np.float32)

    for level in noise_levels:
        # 生成高斯噪声
        noise = np.random.normal(0, level, img.shape).astype(np.float32)

        # 添加噪声
        noisy_img = np.clip(img_float + noise, 0, 255).astype(np.uint8)

        # 保存添加噪声后的图像
        cv2.imwrite(os.path.join(output_dir, f"image_noise_{level}.png"), noisy_img)

        # 将噪声矩阵转换为可视化图像（归一化到 0–255）
        noise_vis = np.clip((noise - noise.min()) / (noise.max() - noise.min()) * 255, 0, 255).astype(np.uint8)

        # 保存噪声图像
        cv2.imwrite(os.path.join(output_dir, f"noise_map_{level}.png"), noise_vis)

        print(f"✅ 已保存: image_noise_{level}.png 和 noise_map_{level}.png")


add_noise_and_save_maps("color_hr/cyan.png", output_dir="noisy_latent")