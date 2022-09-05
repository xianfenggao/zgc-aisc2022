import os

import torch
from torch import nn
from torch.nn import functional as F

from config.path_config import ckpts


class BaseFaceModel(nn.Module):

    def __init__(self,
                 input_shape,
                 ckpt,
                 device='cuda',
                 mean=0.5,
                 std=0.5,
                 ):
        super(BaseFaceModel, self).__init__()
        self.input_shape = input_shape
        self.ckpt = os.path.join(ckpts, ckpt)
        self.mean = mean
        self.std = std
        self.device = device
        self.backbone = None
        self.name = self.__class__.__name__

    def preprocess(self, x):
        # for an rgb [0, 1.], any shape, (N, C, H, W) images, how to preprocess to feed into model
        assert x.size(1) == 3 or x.size(1) == 1
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(x, self.input_shape, mode='bilinear', align_corners=True)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x, return_logits=False):
        out = self.backbone(self.preprocess(x))
        if return_logits:
            return out
        else:
            norm = torch.norm(out, p=2, dim=1, keepdim=True)
            return torch.div(out, norm)
