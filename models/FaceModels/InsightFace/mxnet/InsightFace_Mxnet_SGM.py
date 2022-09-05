import sys

import cv2
import torch

sys.path.append('../../../')

from thirdparty_pkgs.mxnet_insightface import kit_MXNET_LResNet34E_IR, kit_MXNET_LResNet50E_IR, kit_MXNET_LResNet100E_IR
# from thirdparty_pkgs.mxnet_insightface_bk import kit_MXNET_LResNet34E_IR, kit_MXNET_LResNet50E_IR, kit_MXNET_LResNet100E_IR
from models.FaceModels.Base import BaseFaceModel


class MXNET_LResNet34E_IR_SGM(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/InsightFace_Mxnet/model-r34-amf/kit_MXNET_LResNet34E_IR.npy',
                 mean=0.,
                 std=1 / 255.,
                 ):
        super(MXNET_LResNet34E_IR_SGM, self).__init__(input_shape, ckpt, device, mean, std)
        self.backbone = kit_MXNET_LResNet34E_IR.Kit_LResNet34E_IR(self.ckpt)
        self.backbone.eval().to(device)


class MXNET_LResNet50E_IR_SGM(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/InsightFace_Mxnet/model-r50-am-lfw/kit_MXNET_LResNet50E-IR.npy',
                 mean=0.,
                 std=1 / 255.,
                 ):
        super(MXNET_LResNet50E_IR_SGM, self).__init__(input_shape, ckpt, device, mean, std)
        self.backbone = kit_MXNET_LResNet50E_IR.Kit_LResNet50E_IR(self.ckpt)
        # self.backbone.load_state_dict(torch.load(ckpt))
        self.backbone.eval().to(device)


class MXNET_LResNet100E_IR_SGM(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/InsightFace_Mxnet/model-r100-ii/kit_MXNET_LResNet100E-IR.npy',
                 mean=0.,
                 std=1 / 255.,
                 ):
        super(MXNET_LResNet100E_IR_SGM, self).__init__(input_shape, ckpt, device, mean, std)
        self.backbone = kit_MXNET_LResNet100E_IR.Kit_LResNet100E_IR(self.ckpt)
        # self.backbone.load_state_dict(torch.load(ckpt))
        self.backbone.eval().to(device)


if __name__ == "__main__":
    model = MXNET_LResNet100E_IR()
    img = '/home/gaoxianfeng/workspace/FEP/Tdata/group3/aligned/xf.png'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = '/home/gaoxianfeng/workspace/FEP/Tdata/group3/aligned/xf_mouse_masked.png'
    # img = 'Tdata/demo/aligned/goodfellow_aligned.png'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))
