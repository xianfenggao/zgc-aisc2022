from models.FaceModels.Base import BaseFaceModel
from thirdparty_pkgs.TFace.backbone.model_efficientnet_advprop import EfficientNetB0 as EfficientNetB0_ap
from thirdparty_pkgs.TFace.backbone.model_irse_ap import *


class TFace_IR_18_AdvProp_faceemore(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/advprop/ir18_faceemore_epoch10.pth'):
        super(TFace_IR_18_AdvProp_faceemore, self).__init__(input_shape, ckpt, device)
        self.backbone = IR_AdvProp_18(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class TFace_EfficientNetB0_AdvProp_faceemore(BaseFaceModel):
    def __init__(self, device='cuda', input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/TFace/advprop/EfficientNetB0-ap-epoch20'):
        super(TFace_EfficientNetB0_AdvProp_faceemore, self).__init__(input_shape, ckpt, device)
        self.backbone = EfficientNetB0_ap(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)
