import torch
import pytest
from basicsr.archs import build_network
from copy import deepcopy

def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """

    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            logger.info('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)

def registration_model():
    pth_path = "./net_g_latest.pth"
    model1 = torch.load(pth_path)

    network_g = {
        "type": "StyleResNet",
        "num_in_ch": 12,
        "num_out_ch": 6,
        "num_feat": 64,
        "num_block": 8
    }

    net_g = build_network(network_g)
    load_network(net_g, "./net_g_latest.pth", strict=True, param_key='params_ema')
    pass

registration_model()