# https://github.com/timesler/facenet-pytorch
# just pip install facenet-pytorch and place the ckpts in ~/.cache/torch/checkpoint

import cv2
import torch

from models.FaceModels.Base import BaseFaceModel
from thirdparty_pkgs.FaceNet import InceptionResnetV1


class FaceNet_casia(BaseFaceModel):
    def __init__(self, device='cuda',
                 input_shape=(160, 160),
                 ckpt='FaceModels/Recognition/FaceNet/20180408-102900-casia-webface.pt'):
        super(FaceNet_casia, self).__init__(input_shape, ckpt, device)
        self.backbone = InceptionResnetV1(num_classes=10575)
        state_dict = torch.load(self.ckpt, map_location=device)
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.to(device).eval()


class FaceNet_vggface2(BaseFaceModel):

    def __init__(self, device='cuda',
                 input_shape=(160, 160),
                 ckpt='FaceModels/Recognition/FaceNet/20180402-114759-vggface2.pt'
                 ):
        super(FaceNet_vggface2, self).__init__(input_shape, ckpt, device)
        self.backbone = InceptionResnetV1(num_classes=8631)
        self.backbone.load_state_dict(torch.load(self.ckpt, map_location=device), strict=False)
        self.backbone.to(device).eval()


if __name__ == "__main__":
    model = FaceNet_casia()
    img = 'Tdata/1.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img1 = torch.cuda.FloatTensor(img).unsqueeze(0)
    # print(img.shape)

    img = 'Tdata/2.JPG'
    img = cv2.imread(img)[:, :, ::-1].copy().transpose([2, 0, 1])
    img2 = torch.cuda.FloatTensor(img).unsqueeze(0)

    embedding = model(torch.cat([img1, img2], 0))

    print(torch.sum(embedding[0] * embedding[1]))
