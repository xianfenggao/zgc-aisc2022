# https://github.com/ShawnXYang/Face-Robustness-Benchmark

import sys

sys.path.append('../../')

import torch

from models.FaceModels.Base import BaseFaceModel
from thirdparty_pkgs.Face_Robustness_Benchmark_Pytorch.networks.CosFace import sphere
from thirdparty_pkgs.Face_Robustness_Benchmark_Pytorch.networks.ArcFace import IR_50
from thirdparty_pkgs.Face_Robustness_Benchmark_Pytorch.networks.Mobilenet import MobileNet
from thirdparty_pkgs.Face_Robustness_Benchmark_Pytorch.networks.ResNet import ResNet_50
from thirdparty_pkgs.Face_Robustness_Benchmark_Pytorch.networks.SphereFace import sphere20a
from thirdparty_pkgs.Face_Robustness_Benchmark_Pytorch.networks.ShuffleNet import ShuffleNet


class FRB_CosFace(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 96), ckpt='FaceModels/Recognition/Face_Robustness_Benchmark_Pytorch/cosface.pth'):
        super(FRB_CosFace, self).__init__(input_shape, ckpt, device)
        self.backbone = sphere()
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class FRB_ArcFace_IR_50(BaseFaceModel):
    def __init__(self,
                 input_shape=(112, 112),
                 ckpt="FaceModels/Recognition/Face_Robustness_Benchmark_Pytorch/model_ir_se50.pth",
                 device='cuda',
                 ):
        super(FRB_ArcFace_IR_50, self).__init__(input_shape, ckpt, device)
        self.backbone = IR_50(input_shape)
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class FRB_MobileNet(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/Face_Robustness_Benchmark_Pytorch/Backbone_Mobilenet_Epoch_125_Batch_710750_Time_2019-04-14-18-15_checkpoint.pth'):
        super(FRB_MobileNet, self).__init__(input_shape, ckpt, device)
        self.backbone = MobileNet(2)
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class FRB_ResNet50(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/Face_Robustness_Benchmark_Pytorch/Backbone_ResNet_50_Epoch_36_Batch_204696_Time_2019-04-14-14-44_checkpoint.pth'):
        super(FRB_ResNet50, self).__init__(input_shape, ckpt, device)
        self.backbone = ResNet_50(input_shape)
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class FRB_SphereFace(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 96),
                 ckpt='FaceModels/Recognition/Face_Robustness_Benchmark_Pytorch/sphere20a_20171020.pth'):
        super(FRB_SphereFace, self).__init__(input_shape, ckpt, device)
        self.backbone = sphere20a()
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class FRB_ShuffleNetV1(BaseFaceModel):

    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/Face_Robustness_Benchmark_Pytorch/Backbone_ShuffleNet_Epoch_124_Batch_1410128_Time_2019-05-05-02-33_checkpoint.pth'):
        super(FRB_ShuffleNetV1, self).__init__(input_shape, ckpt, device)
        self.backbone = ShuffleNet(pooling='GDConv')
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)
