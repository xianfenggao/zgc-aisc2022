import torch

from models.FaceModels.Base import BaseFaceModel
from thirdparty_pkgs.TFace.backbone import *


class TFace_IR_18_glintasia(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_18_glintasia/Backbone_Epoch_26_checkpoint.pth'):
        super(TFace_IR_18_glintasia, self).__init__(input_shape, ckpt, device)
        self.backbone = IR_18(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_MobileFaceNet_face_emore(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet_faceemore/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_MobileFaceNet_face_emore, self).__init__(input_shape, ckpt, device)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_MobileFaceNet_glintasia(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet_glintasia/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_MobileFaceNet_glintasia, self).__init__(input_shape, ckpt, device)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


######################## IR_SE_50 Model Zoo ################


class TFace_IR_SE_50_2data(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/2data/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_2data, self).__init__(device=device,
                                                   input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_5data(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/5data/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_5data, self).__init__(device=device,
                                                   input_shape=input_shape,
                                                   ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_faceemore(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/faceemore/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_faceemore, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_faceglint(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/faceglint/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_faceglint, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_glintasia(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/glintasia/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_glintasia, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_umd(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/umd/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_umd, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_vggface2(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/vggface2/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_vggface2, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_webface(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/webface/Backbone_Epoch_15_checkpoint.pth'):
        super(TFace_IR_SE_50_webface, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR_SE_50_withaug_faceemore(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_SE_50/faceemore/Backbone_Epoch_33_checkpoint_withaug.pth'):
        super(TFace_IR_SE_50_withaug_faceemore, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_SE_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


######################## MobileFaceNet ZOO ################


class TFace_MobileFaceNet_Aug_faceemore(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet/faceemore/Backbone_Epoch_35_checkpoint.pth'):
        super(TFace_MobileFaceNet_Aug_faceemore, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_MobileFaceNet_Aug_glint360k(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet/glint360k/Backbone_Epoch_25_checkpoint.pth'):
        super(TFace_MobileFaceNet_Aug_glint360k, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_MobileFaceNet_Aug_glintasia(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet/glintasia/Backbone_Epoch_35_checkpoint.pth'):
        super(TFace_MobileFaceNet_Aug_glintasia, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_MobileFaceNet_Aug_umd(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet/umd/Backbone_Epoch_35_checkpoint.pth'):
        super(TFace_MobileFaceNet_Aug_umd, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_MobileFaceNet_Aug_vggface2(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet/vggface2/Backbone_Epoch_35_checkpoint.pth'):
        super(TFace_MobileFaceNet_Aug_vggface2, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_MobileFaceNet_Aug_webface(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/MobileFaceNet/webface/Backbone_Epoch_35_checkpoint.pth'):
        super(TFace_MobileFaceNet_Aug_webface, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = MobileFaceNet(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


######################## IR 101 MODEL ZOO ################

class TFace_IR101_Aug_faceemore(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_101/faces_emore/Backbone_Epoch_20_checkpoint.pth'):
        super(TFace_IR101_Aug_faceemore, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_101(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_IR101_Aug_glint360k(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/IR_101/deepglint_360k/Backbone_Epoch_7_checkpoint.pth'):
        super(TFace_IR101_Aug_glint360k, self).__init__(device=device, input_shape=input_shape, ckpt=ckpt)
        self.backbone = IR_101(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)
