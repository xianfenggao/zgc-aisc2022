"""
https://github.com/biubug6/Pytorch_Retinaface

we do not implement the nms so that it works only for detecting one face

"""

import os
import sys
# sys.data_path.append('thirdparty_pkgs/Retinaface_Pytorch/models')

import numpy as np
import torch
import torchvision
import cv2

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from thirdparty_pkgs.Retinaface_Pytorch.models.retinaface import RetinaFace
from thirdparty_pkgs.Retinaface_Pytorch.layers.functions.prior_box import PriorBox
from thirdparty_pkgs.Retinaface_Pytorch.utils.box_utils import decode, decode_landm
from thirdparty_pkgs.Retinaface_Pytorch.utils.nms.py_cpu_nms import py_cpu_nms
from config.path_config import ckpts
from skimage import transform as trans
from thirdparty_pkgs.WarpAffine2GridSample.utils import transform_keypoints
from thirdparty_pkgs.Retinaface_Pytorch.data.config import cfg_mnet, cfg_re50

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)


class BaseRetinaFace(nn.Module):
    def __init__(self, device):
        super(BaseRetinaFace, self).__init__()
        self.device = device
        self.mean = torch.FloatTensor((104, 117, 123)).view(1, 3, 1, 1).to(device)
        self.src = arcface_src
        self.estimator = trans.SimilarityTransform()

    def preprocess(self, x):
        # for an rgb [0, 1], any shape, (N, C, H, W) images, how to preprocess to feed into model
        return x * 255 - self.mean

    def depreprocess(self, x):
        return (x + self.mean) / 255.

    def detect_faces(self, x, size=112):
        grids = self.get_batch_grids(x, size)
        return F.grid_sample(x, grids, mode='bilinear')

    @torch.no_grad()
    def detect_faces_cv2(self, x, size=112):
        x = torch.FloatTensor(x[:, :, ::-1].copy().transpose([2, 0, 1])).unsqueeze(0).to(self.device) / 255.
        face = self.detect_faces(x, size)[0] * 255
        return face.cpu().numpy().transpose([1, 2, 0]).astype('uint8')[:, :, ::-1]

    def get_batch_grids(self, x, target_size):
        '''
        :param x:
        :param size:
        :return:
        '''
        _, _, im_height, im_width = x.size()
        _, _, h, w = x.size()
        if not hasattr(self, 'priors') or (im_height, im_width) != self.priors_size:
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            self.priors = priors
            self.priors_size = (im_height, im_width)

        loc, conf, landms = self.backbone(self.preprocess(x))
        # （1，10752，4） （N，10752，2） （1，10752，10）

        # 返回概率最大的坐标点Tensor
        idx = torch.argmax(conf[:, :, 1], 1)
        thetas = []
        for i, id in enumerate(idx):
            ldm = decode_landm(landms[i], self.priors, self.cfg['variance'])
            ldm = ldm[id].reshape(5, 2).data.cpu().numpy() * np.array([im_width, im_height])[None, :]
            self.estimator.estimate(ldm, self.src * target_size / 112)
            Meta = self.estimator.params
            _src = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
            dst = transform_keypoints(_src, Meta)
            src = _src / [im_width, im_height] * 2 - 1
            dst = dst / [target_size, target_size] * 2 - 1
            theta = trans.estimate_transform("affine", src=dst, dst=src).params
            thetas.append(theta[:2])

        thetas = torch.FloatTensor(thetas).to(x.device)
        grids = F.affine_grid(thetas, size=[x.size(0), x.size(1), target_size, target_size], align_corners=True)
        return grids

    def detect_boxes(self, x):
        '''
        boxes
        :param x:  torch.Tensor  (N, xxx)  for a batch images
        :return:  a batch boxes , (x1, y1, x2, y2) for 2nd dimension
        '''
        _, _, im_height, im_width = x.size()
        _, _, h, w = x.size()
        if not hasattr(self, 'scale') or (im_height, im_width) != self.priors_size:
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            self.priors = priors
            self.priors_size = (im_height, im_width)
            self.scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)

        loc, conf, landms = self.backbone(self.preprocess(x))
        # （1，10752，4） （N，10752，2） （1，10752，10）

        # 返回概率最大的坐标点Tensor
        idx = torch.argmax(conf[:, :, 1], 1)
        boxes = []
        for i, id in enumerate(idx):
            box = decode(loc[i], self.priors, self.cfg['variance'])[id]
            box = (box * self.scale).long()
            box = torch.where(box < 0, 0, box)
            boxes.append(box)
            # box = torch.where(box < 0, 0, box)
            # x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # res.append(resizer(x[i:i+1, :, y1:y2, x1:x2]))
        return torch.stack(boxes)

    def get_warp_matrix(self, x, target_size=112):
        '''
        get opencv warpAffine matrix for a single image used for align
        :param x: an opencv bgr image
        :return:
        '''
        x = torch.FloatTensor(x[:, :, ::-1].copy().transpose([2, 0, 1])).unsqueeze(0).to(self.device) / 255.
        _, _, im_height, im_width = x.size()
        _, _, h, w = x.size()
        if not hasattr(self, 'priors') or (im_height, im_width) != self.priors_size:
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            self.priors = priors
            self.priors_size = (im_height, im_width)

        loc, conf, landms = self.backbone(self.preprocess(x))
        # （1，10752，4） （N，10752，2） （1，10752，10）

        # 返回概率最大的坐标点Tensor
        idx = torch.argmax(conf[:, :, 1], 1)
        for i, id in enumerate(idx):
            ldm = decode_landm(landms[i], self.priors, self.cfg['variance'])
            ldm = ldm[id].reshape(5, 2).data.cpu().numpy() * np.array([im_width, im_height])[None, :]
            self.estimator.estimate(ldm, self.src * target_size / 112)
            Meta = self.estimator.params
            return Meta[:2]


class RetinaFace_MobileNet(BaseRetinaFace):
    def __init__(self, device='cuda', ckpt=os.path.join(ckpts, 'FaceModels/Detection/Retinaface_Pytorch/mobilenet0.25_Final.pth')):
        super(RetinaFace_MobileNet, self).__init__(device)
        self.backbone = RetinaFace(cfg=cfg_mnet, phase='test')
        self.backbone.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        self.backbone.eval().to(device)
        self.cfg = cfg_mnet
        self.name = 'RetinaFace_mobilenet025'


class RetinaFace_ResNet50(BaseRetinaFace):
    def __init__(self, device='cuda', ckpt=os.path.join(ckpts, 'FaceModels/Detection/Retinaface_Pytorch/Resnet50_Final.pth')):
        super(RetinaFace_ResNet50, self).__init__(device)
        self.backbone = RetinaFace(cfg=cfg_re50, phase='test')
        self.backbone.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        self.backbone.eval().to(device)
        self.cfg = cfg_re50
        self.name = 'RetinaFace_ResNet50'

