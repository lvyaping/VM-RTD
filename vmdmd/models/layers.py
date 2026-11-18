"""Small timm-compatible layer utilities; state-dict structure is unchanged."""
import torch
from torch import nn
from torch.nn.init import trunc_normal_


def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device,
                           dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep
