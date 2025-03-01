from basicsr.data.taco_dataset import TacoDataset, TacoSplitDataset
import torch
import time
import tacoreader
import rasterio as rio
from matplotlib import pyplot as plt
import torchvision.utils as vutils
import cv2

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

def imwrite(im_in, path, chn='rgb', dtype_in='float32', qf=None):
    '''
    Save image.
    Input:
        im: h x w x c, numpy tensor
        path: the saving path
        chn: the channel order of the im,
    '''
    from skimage.util import img_as_ubyte, img_as_float32
    from pathlib import Path
    im = im_in.copy()
    if isinstance(path, str):
        path = Path(path)
    if dtype_in != 'uint8':
        im = img_as_ubyte(im)

    flag = cv2.imwrite(str(path), im[:,:,[2,1,0]])

    return flag

def save_tensor(im_tensor, path):
    im_tensor = vutils.make_grid(im_tensor, nrow=4, normalize=True, scale_each=True) # c x H x W
    im_np = im_tensor.cpu().permute(1,2,0).numpy()
    imwrite(im_np, path)

def test_train():
    opt = {
        "taco_paths": "/mnt/code/deep_learning/BasicSR/datasets/SEN2NAIPv2/sen2naipv2-crosssensor.taco",
        "split_percent": [0.9, 0.09, 0.01],
        "split": 1,
        "phase": "train",
        "scale": 4,
        "gt_size": 256,
        "use_hflip": True,
        "use_rot": True,
    }
    dataloader = TacoSplitDataset(opt)
    datamodule = torch.utils.data.DataLoader(dataloader, batch_size=4, shuffle=False)
    start = time.time()
    batch = next(iter(datamodule))
    gt_tensor = batch["gt"]
    save_tensor(gt_tensor, "resshift2.png")
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

def test_skimage():
    import numpy as np
    from skimage.util import img_as_ubyte
    h, w = 5, 5  # Specify the height and width of the array
    arr = np.random.rand(h, w, 4)  # Create a random array with shape (h, w, 4)

    # Assign random values in specified ranges to each band
    arr[:, :, 0] = np.random.randint(0, 11, size=(h, w))  # Values in range [0, 10]
    arr[:, :, 1] = np.random.randint(10, 21, size=(h, w))  # Values in range [10, 20]
    arr[:, :, 2] = np.random.randint(20, 31, size=(h, w))  # Values in range [20, 30]
    arr[:, :, 3] = np.random.randint(31, 41, size=(h, w))
    arr = arr/100
    arr2 = img_as_ubyte(arr)
    pass

def test_color2():
    import cv2
    single_band = cv2.imread("resshift2.png", 0)
    cv2.imwrite("gray.png", single_band)
    #pseudo_color_image = cv2.applyColorMap(single_band, cv2.COLORMAP_RAINBOW)
    cv2.imwrite("color.png", pseudo_color_image)


test_color2()