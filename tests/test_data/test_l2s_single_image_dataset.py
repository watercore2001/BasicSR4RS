from basicsr.data.l2s_single_dataset import L2SSingleDataset, L2SSingleSplitDataset
from basicsr.utils.img_util import minusone_one_tensor_to_ubyte_numpy
import os

from skimage.io import imsave


def save_tensor_dict_as_png_skimage(data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for domain_name, band_dict in data_dict.items():
        if isinstance(band_dict, dict):
            for band_key, tensor in band_dict.items():
                np_image = minusone_one_tensor_to_ubyte_numpy(tensor)
                filename = f"{domain_name}_{band_key}.png"
                save_path = os.path.join(output_dir, filename)
                imsave(save_path, np_image)
                print(f"Saved (skimage): {save_path}")


def xials():
    opt = {
        "root_path": "/mnt/code/deep_learning/BasicSR/data/coreg",
        "phase": "train",
        "psnr_thresh": 22,
        "ssim_thresh": 0.8,
        "gt_size": 384,
        "use_hflip": True,
        "use_rot": True,
    }

    debug_dataset = L2SSingleDataset(opt=opt)
    print(len(debug_dataset))
    i = 0
    for sample in debug_dataset:
        i += 1
        if i > 10:
            break
        save_tensor_dict_as_png_skimage(sample, f"visual/l2s_single_image/{i}")


def split():
    opt = {
        "root_path": "/mnt/code/deep_learning/BasicSR/data/coreg",
        "phase": "train",
        "psnr_thresh": 22,
        "ssim_thresh": 0.8,
        "gt_size": 288,
        "use_hflip": True,
        "use_rot": True,
        "split_percent": [ 0.99, 0.01 ],
        "split": 1,
    }

    debug_dataset = L2SSingleSplitDataset(opt=opt)
    print(len(debug_dataset))
    for sample in debug_dataset:
        print(1)


if __name__ == '__main__':
    xials()
