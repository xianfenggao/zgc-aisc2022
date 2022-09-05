# https://github.com/deepinsight/insightface

import cv2
import torch

from models.FaceModels.Base import BaseFaceModel
from thirdparty_pkgs.insightFace.backbones.iresnet import iresnet50, iresnet200


class ArcFace_torch_ir50_faceemore(BaseFaceModel):

    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/InsightFace_MYtranined/ms1mv3_arcface_r50/backbone.pth'):
        super(ArcFace_torch_ir50_faceemore, self).__init__(input_shape, ckpt, device)
        self.backbone = iresnet50()
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


class ArcFace_torch_ir200_faceemore(BaseFaceModel):

    def __init__(self, device='cuda',
                 input_shape=(112, 112),
                 ckpt='FaceModels/Recognition/InsightFace_MYtranined/ms1mv3_arcface_r200/backbone.pth'):
        super(ArcFace_torch_ir200_faceemore, self).__init__(input_shape, ckpt, device)
        self.backbone = iresnet200()
        self.backbone.feature = True
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.backbone.eval().to(device)


if __name__ == "__main__":
    model = ArcFace()
    img = 'Tdata/demo/aligned/03.png'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = 'Tdata/demo/aligned/huangxiang.jpg'
    # img = 'Tdata/demo/aligned/goodfellow_aligned.png'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))
