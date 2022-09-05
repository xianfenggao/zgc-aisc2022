# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch

import sys

import cv2

sys.path.append('../')

import torch

from thirdparty_pkgs.face_evoLVe_PyTorch import IR_50, IR_152
from models.FaceModels.Base import BaseFaceModel


class evoLVe_IR_50(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/face_evoLVe_PyTorch/ms1m-ir50/backbone_ir50_ms1m_epoch63.pth'):
        super(evoLVe_IR_50, self).__init__(input_shape, ckpt, device)
        self.backbone = IR_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class evoLVe_IR_50_Asia(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/face_evoLVe_PyTorch/bh-ir50/backbone_ir50_asia.pth'):
        super(evoLVe_IR_50_Asia, self).__init__(input_shape, ckpt, device)
        self.backbone = IR_50(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class evoLVe_IR_152(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/face_evoLVe_PyTorch/ms1m-ir152/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'):
        super(evoLVe_IR_152, self).__init__(input_shape, ckpt, device)
        self.backbone = IR_152(input_shape)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


if __name__ == "__main__":
    model = evoLVe_IR_152()
    img = 'Tdata/attacked.png'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = 'Tdata/target.png'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))
