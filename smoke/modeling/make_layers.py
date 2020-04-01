import math

import torch
from torch import nn

from smoke.config import cfg


def _make_conv_level(in_channels, out_channels, num_convs, norm_func,
                     stride=1, dilation=1):
    """
    make conv layers based on its number.
    """
    modules = []
    for i in range(num_convs):
        modules.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride if i == 0 else 1,
                      padding=dilation, bias=False, dilation=dilation),
            norm_func(out_channels),
            nn.ReLU(inplace=True)])
        in_channels = out_channels

    return nn.Sequential(*modules)


def group_norm(out_channels):
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)


def _fill_up_weights(up):
    # todo: we can replace math here?
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def _fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
