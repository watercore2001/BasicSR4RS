from torch.utils.data import DataLoader
from os import path as osp
from basicsr.data.taco_dataset import TacoSplitDataset
from basicsr.archs.autoencoder_arch import VQModelTorch
import torch
from basicsr.utils import imwrite, zero_one_tensor_to_ubyte_numpy
import numpy as np
import os
import random
import time
import torch
from os import path as osp
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from basicsr.train import build_dataset, build_dataloader

dataset_opt = {
    "name": "S2N_hismatch_train",
    "type": "TacoSplitDataset",
    "taco_paths": [
        "data/SEN2NAIPv2/sen2naipv2-histmatch.0000.part.taco",
        "data/SEN2NAIPv2/sen2naipv2-histmatch.0001.part.taco",
        "data/SEN2NAIPv2/sen2naipv2-histmatch.0002.part.taco",
        "data/SEN2NAIPv2/sen2naipv2-histmatch.0003.part.taco"
    ],
    "split_percent": [0.999, 0.001],
    "split": 1,
    "phase": "val",
    "scale": 4,
    "gt_size": 256,
    "band_idx": [1,2,3]
}


val_set1 = build_dataset(dataset_opt)
val_set2 = build_dataset(dataset_opt)
val_dataloader = build_dataloader(val_set2, dataset_opt, num_gpu=1, seed=seed)

device = "cuda"

model = VQModelTorch(
        embed_dim=3,
        n_embed=8192,
        ddconfig={
            "double_z": False,
            "z_channels" :3,
            "resolution" :256,
            "in_channels" :3,
            "out_ch" :3,
            "ch" :128,
            "ch_mult" : [1, 2, 4],
            "num_res_blocks" : 2,
            "attn_resolutions" : [],
            "dropout":0.0,
            "padding_mode" :"zeros"}).to(device)

params = torch.load("experiments/pretrained_models/autoencoder_vq_f4.pth", map_location=device)

model.load_state_dict(params)
model.eval()

visual_fodler = "ldm_vq"


def rswrite( folder: str, filename_dict: dict):
    for key, value in filename_dict.items():
        save_file_path = osp.join(folder, f'{key}.png')
        if not osp.exists(save_file_path) and value is not None:
            imwrite(value, save_file_path)

from torchvision.transforms import Normalize

def norm(img: torch.Tensor):
    # 0-1 B RGB+NIR H W
    img = img[:,0:3,:,:]
    #img = torch.clamp(img,0,1)
    # norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # img = norm(img)
    return img

def get_ubyte(img):
    img = torch.clamp(img, -1, 1)
    img = img.detach().cpu()
    img = (img+1)/2
    return zero_one_tensor_to_ubyte_numpy(img)

with torch.no_grad():
    for batch in val_dataloader:

        lq_img = norm(batch['lq'].to(device))
        gt_img = norm(batch['gt'].to(device))
        lq_ae = model(lq_img)
        gt_ae = model(gt_img)

        lq_img = get_ubyte(lq_img)  # numpy H(BW)(RGBNIR) uint8
        gt_img = get_ubyte(gt_img)  # numpy H(BW)(RGBNIR) uint8
        lq_ae = get_ubyte(lq_ae)  # numpy H(BW)(RGBNIR) uint8
        gt_ae = get_ubyte(gt_ae)

        lq_path = batch['lq_path'][0]
        if lq_path.endswith('.taco'):
            img_name = osp.basename(lq_path.split(',')[0])
        print(img_name)
        rgb_path = osp.join(visual_fodler, "RGB", img_name)
        rgb_dict = {
            "lq": lq_img[..., [2, 1, 0]],
            "gt": gt_img[..., [2, 1, 0]] if gt_img is not None else None,
            f"lq_ae": lq_ae[..., [2, 1, 0]],
            f"gt_ae": gt_ae[..., [2, 1, 0]],
        }

        rswrite(rgb_path, rgb_dict)