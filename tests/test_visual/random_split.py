from basicsr.train import build_dataset, build_dataloader
from os import path as osp
import torch
from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info
torch.backends.cudnn.benchmark = True
# random seed
rank, world_size = get_dist_info()


seed = 0
set_random_seed(seed + rank)

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
}

val_set1 = build_dataset(dataset_opt)
val_set2 = build_dataset(dataset_opt)
val_dataloader1 = build_dataloader(val_set2, dataset_opt, num_gpu=1, seed=0)
val_dataloader2 = build_dataloader(val_set2, dataset_opt, num_gpu=1, seed=0)

for batch1, batch2 in zip(val_dataloader1, val_dataloader2):
    lq_path1 = batch1['lq_path'][0]
    if lq_path1.endswith('.taco'):
        img_name1 = osp.basename(lq_path1.split(',')[0])
    lq_path2 = batch1['lq_path'][0]
    if lq_path2.endswith('.taco'):
        img_name2 = osp.basename(lq_path2.split(',')[0])
    assert img_name1 == img_name2
    print(img_name1)