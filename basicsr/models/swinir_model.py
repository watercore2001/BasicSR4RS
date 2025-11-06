import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.models.srrs_model import SRRSModel
from basicsr.models.srrs_l2s_model import L2SSingleModel
from basicsr.models.srrs_l2shm_model import L2SSingleHMModel


@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


@MODEL_REGISTRY.register()
class SwinIRRSModel(SwinIRModel, SRRSModel):
    pass


@MODEL_REGISTRY.register()
class SwinIRL2sModel(SwinIRModel, L2SSingleModel):
    pass


@MODEL_REGISTRY.register()
class SwinIRHMModel(L2SSingleHMModel):
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g'].get('sr_net_args', {}).get('window_size')
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        def crop(img):
            return img[:, :, 0:img.shape[2] - mod_pad_h * scale, 0:img.shape[3] - mod_pad_w * scale]

        self.output = {
            key: crop(img)
            for key, img in self.output.items()
        }
