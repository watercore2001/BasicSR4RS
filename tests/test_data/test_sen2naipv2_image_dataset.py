from basicsr.data.taco_dataset import TacoDataset, TacoSplitDataset
import torch
import time
import tacoreader
import rasterio as rio
from matplotlib import pyplot as plt

def test_cross_sensor():
    path = "/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-crosssensor.taco"
    dataloader = TacoDataset(path)
    datamodule = torch.utils.data.DataLoader(dataloader, batch_size=100, shuffle=False)
    start = time.time()
    batch = next(iter(datamodule))
    end = time.time()
    elapsed = end - start
    assert elapsed < 1
    pass

def test_unet():
    paths = ["/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-histmatch.0000.part.taco",
             "/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-histmatch.0001.part.taco",
             "/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-histmatch.0002.part.taco",
             "/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-histmatch.0003.part.taco"]
    dataloader = TacoDataset(paths)
    datamodule = torch.utils.data.DataLoader(dataloader, batch_size=100, shuffle=False)
    start = time.time()
    batch = next(iter(datamodule))
    end = time.time()
    elapsed = end - start
    assert elapsed < 10
    pass


def test_train():
    opt = {
        "taco_paths": "/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-crosssensor.taco",
        "phase": "val",
        "scale": 4,
        "gt_size": 512,
        "use_hflip": True,
        "use_rot": True,
    }
    dataloader = TacoSplitDataset(opt)
    datamodule = torch.utils.data.DataLoader(dataloader, batch_size=100, shuffle=False)
    start = time.time()
    batch = next(iter(datamodule))
    end = time.time()
    elapsed = end - start
    assert elapsed < 1
    pass

def test_color():
    path = "/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-crosssensor.taco"
    dataset = tacoreader.load(path)
    # Read a sample
    for sample_idx in [300, 4400, 500, 600, 700, 800, 900, 1000]:
        lr = dataset.read(sample_idx).read(0)
        hr = dataset.read(sample_idx).read(1)

        # Retrieve the data
        with rio.open(lr) as src, rio.open(hr) as dst:
            lr_data = src.read([1, 2, 3], window=rio.windows.Window(0, 0, 256 // 4, 256 // 4))
            hr_data = dst.read([1, 2, 3], window=rio.windows.Window(0, 0, 256, 256))

        # Display
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
        ax1.imshow(lr_data.transpose(1, 2, 0) / 3000)
        ax1.set_title("Low Resolution - Sentinel 2")
        ax2.imshow(hr_data.transpose(1, 2, 0) / 3000)
        ax2.set_title("High Resolution - NAIP")
        ax3.imshow(lr_data.transpose(1, 2, 0) / 2000)
        ax3.set_title("Low Resolution - Sentinel 2")
        ax4.imshow(hr_data.transpose(1, 2, 0) / 2000)
        ax4.set_title("High Resolution - NAIP")
        plt.show()

test_color()