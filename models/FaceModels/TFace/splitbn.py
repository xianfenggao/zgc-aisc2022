import torch

from models.FaceModels.Base import BaseFaceModel
from thirdparty_pkgs.TFace.backbone import *
from thirdparty_pkgs.TFace.util.split_batchnorm import convert_splitbn_model


class TFace_IR_SE_50_splitbn_2data(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/splitbn/2data/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_splitbn_2data, self).__init__(input_shape, ckpt, device)
        model = IR_SE_50(input_shape)
        self.backbone = convert_splitbn_model(model, 2)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_splitbn_alldata(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112), ckpt='FaceModels/Recognition/TFace/splitbn/alldata/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_splitbn_alldata, self).__init__(input_shape, ckpt, device)
        model = IR_SE_50(input_shape)
        self.backbone = convert_splitbn_model(model, 7)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_DenseNet201_splitbn_faceemore_glintasia(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/DenseNet/DenseNet201_faceemore_glintasia/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_DenseNet201_splitbn_faceemore_glintasia, self).__init__(device=device,
                                                                            input_shape=input_shape, ckpt=ckpt)
        model = densenet201(input_shape)
        self.backbone = convert_splitbn_model(model, 2)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_DenseNet201_splitbn_faceemore_glintasia_ImageNetAutoAugment(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/DenseNet/DenseNet201_faceemore_glintasia_ImageNetAutoAugment/Backbone_Epoch_11_checkpoint.pth'):
        super(TFace_DenseNet201_splitbn_faceemore_glintasia_ImageNetAutoAugment, self).__init__(device=device,
                                                                                                input_shape=input_shape,
                                                                                                ckpt=ckpt)
        model = densenet201(input_shape)
        self.backbone = convert_splitbn_model(model, 2)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)
