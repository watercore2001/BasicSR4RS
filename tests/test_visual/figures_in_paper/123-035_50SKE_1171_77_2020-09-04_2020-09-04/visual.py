import os
from PIL import Image

def get_image_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_reference_size(image_paths, folder, exclude_name='lq.png'):
    for fname in image_paths:
        if fname != exclude_name:
            with Image.open(os.path.join(folder, fname)) as img:
                return img.size
    raise ValueError("未找到参考图像尺寸")

def resize_if_needed(image, target_size):
    if image.size != target_size:
        return image.resize(target_size, Image.BICUBIC)
    return image

def crop_image(image, x, y, crop_ratio):
    width, height = image.size
    crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
    crop_x = min(x, width - crop_w)
    crop_y = min(y, height - crop_h)
    return image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))

def process_and_save_images(input_dir, output_dir, x, y, crop_ratio=0.3):
    """
    左上角为原点，x向右，y向下
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = get_image_files(input_dir)
    ref_size = get_reference_size(image_files, input_dir)

    for fname in image_files:
        path = os.path.join(input_dir, fname)
        with Image.open(path) as img:
            img = resize_if_needed(img, ref_size)
            cropped = crop_image(img, x, y, crop_ratio)
            cropped.save(os.path.join(output_dir, fname))

    print(f"✅ 所有图像已处理并保存至：{output_dir}")

def process_folder(folder, x, y):

    process_and_save_images(
        input_dir=f'{folder}/rgb',
        output_dir=f'{folder}/rgb_crop',
        x=x,
        y=y
    )
    process_and_save_images(
        input_dir=f'{folder}/nss',
        output_dir=f'{folder}/nss_crop',
        x=x,
        y=y
    )

process_folder("/mnt/code/deep_learning/BasicSR/tests/test_visual/figures_in_paper/123-035_50SKE_1171_77_2020-09-04_2020-09-04",
               x=80, y=100)

process_folder("/mnt/code/deep_learning/BasicSR/tests/test_visual/figures_in_paper/123-035_50SKE_1174_90_2020-09-04_2020-09-04",
               x=60, y=40)

process_folder("/mnt/code/deep_learning/BasicSR/tests/test_visual/figures_in_paper/124-035_49SGV_1167_230_2021-09-30_2021-09-29",
               x=100, y=20)
