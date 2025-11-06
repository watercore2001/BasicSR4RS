from basicsr.data.l2s_single_dataset import filter_metrics

folder = "data/single_square"
psnr_min = 24
ssim_min = 0.8
psnr_max = 38

filter_metrics(folder, psnr_min, ssim_min, psnr_max)