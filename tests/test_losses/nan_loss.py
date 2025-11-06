import torch

# bad_loss = "/mnt/download/meta_iter_10050.pt"
#
# data = torch.load(bad_loss)
# print(data)

def fuck():

    x = torch.tensor([1.0, float('nan'), 3.0], requires_grad=True)
    x_detached = x.detach()  # ✅ 不会报错
    print(x_detached)
    torch.save(x_detached, 'x.pt')
    data = torch.load('x.pt')
    pass


import torch

import torch
import torch.nn.functional as F

def inspect_nan_regions(out, gt, lq, window=2, max_print=5):
    """
    检查 out 张量中的 NaN，并打印每个 NaN 的位置及其周围像素值，
    同时打印对应位置的 gt 和插值后的 lq 区域。

    参数:
        out (torch.Tensor): 模型输出，形状 (B, C, H, W)
        gt (torch.Tensor): Ground Truth，形状与 out 相同
        lq (torch.Tensor): 输入图像，形状可能不同，将插值到与 out 相同
        window (int): 周围区域的半径，window=2 表示打印 5x5 区域
        max_print (int): 最多打印多少个 NaN 点的信息
    """
    assert out.shape == gt.shape, "out 和 gt 的形状必须一致"
    B, C, H, W = out.shape

    # 插值 lq 到目标大小
    if lq.shape[2:] != (H, W):
        lq_resized = F.interpolate(lq, size=(H, W), mode='bilinear', align_corners=False)
    else:
        lq_resized = lq

    nan_mask = torch.isnan(out)
    nan_indices = nan_mask.nonzero()

    total_nan = nan_indices.shape[0]
    print(f"发现 {total_nan} 个 NaN")

    if total_nan == 0:
        return

    for idx in nan_indices[:max_print]:
        b, c, h, w = idx.tolist()
        print(f"\n🧨 NaN 位置: batch={b}, channel={c}, h={h}, w={w}")

        h_start = max(h - window, 0)
        h_end = min(h + window + 1, H)
        w_start = max(w - window, 0)
        w_end = min(w + window + 1, W)

        out_patch = out[b, c, h_start:h_end, w_start:w_end].detach().cpu()
        gt_patch = gt[b, c, h_start:h_end, w_start:w_end].detach().cpu()
        lq_patch = lq_resized[b, c, h_start:h_end, w_start:w_end].detach().cpu()

        print("🔍 模型输出 (out) 周围像素值：")
        print(out_patch)
        print("🎯 Ground Truth (gt) 对应区域：")
        print(gt_patch)
        print("📥 插值后的输入 (lq) 对应区域：")
        print(lq_patch)

def out_test():
    def check_nan(tensor, name):
        has_nan = torch.isnan(tensor).any().item()
        print(f"{name} contains NaN: {has_nan}")
    lq_path = "17609/lq_iter_17609.pt"
    out_path = "17609/out_iter_17609.pt"
    gt_path = "17609/gt_iter_17609.pt"
    meta_path = "17609/meta_iter_17609.pt"

    meta = torch.load(meta_path)
    lq = torch.load(lq_path)
    out = torch.load(out_path)
    gt = torch.load(gt_path)
    check_nan(lq, "lq")
    check_nan(out, "out")
    check_nan(gt, "gt")
    inspect_nan_regions(out, gt, lq)
    print(meta)
    pass

out_test()