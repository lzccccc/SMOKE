import torch

from .smoke_head.smoke_head import build_smoke_head


def build_heads(cfg, in_channels):
    if cfg.MODEL.SMOKE_ON:
        return build_smoke_head(cfg, in_channels)
