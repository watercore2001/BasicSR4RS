import torch
# from src.lanczos import lanczos_shift, lanczos_kernel
from basicsr.losses.registered_loss import RegisteredLoss, ShiftConv2d
from torch.nn import MSELoss

def registeredLoss():
    d = 20
    img1 = torch.zeros((3, d, d))
    img1[[0], :, d//2] = 1
    img1[[1], d//2, :] = 1
    img2 = torch.zeros(3, d, d)
    img2[0, :, :] = .5 * torch.eye(d).fliplr()
    img2[1, :, :] = .5 * torch.eye(d)
    x = torch.stack([img1, img2])

    w = 2
    start, end, step = -(w/2), w/2, 0.5
    reg_mse = RegisteredLoss(start, end, step=step, loss_func="l1")
    x_shifted = reg_mse._shiftconv2d(x)
    assert x_shifted.shape == (2, 25, 3, d, d)

    y = torch.cat([x_shifted[0, [3]], x_shifted[1, [15]]])
    loss_all_shifts = reg_mse._shifted_loss(x, y).detach()
    assert loss_all_shifts.shape == (2, 25)

    mes_loss = MSELoss(reduction='none')
    print(mes_loss(x, y).mean((-3,-2,-1)))
    print(reg_mse(x, y))
    assert (reg_mse(x, y) == 0).all()

registeredLoss()