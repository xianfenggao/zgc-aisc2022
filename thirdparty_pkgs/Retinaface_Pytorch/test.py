import torch
import cv2
import numpy as np
from torch.autograd import Variable
from thirdparty_pkgs.Retinaface_Pytorch.models.retinaface import RetinaFace
from thirdparty_pkgs.Retinaface_Pytorch.layers.functions.prior_box import PriorBox
from thirdparty_pkgs.Retinaface_Pytorch.utils.box_utils import decode, decode_landm
from thirdparty_pkgs.Retinaface_Pytorch.utils.nms.py_cpu_nms import py_cpu_nms

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

net = RetinaFace(cfg=cfg_re50, phase='test')
#
mobileface_dict = torch.load("../../ckpts/Retinaface_Pytorch/Resnet50_Final.pth")
new_dict = {}
for k, v in mobileface_dict.items():
    # print(k[7:])
    k = k[7:]
    new_dict.setdefault(k, v)
# net = load_model(net, args.trained_model, args.cpu)
net.load_state_dict(new_dict)
net.to('cuda')
net.eval()

print('Finished loading model!')

# print(net)

img2 = '../../Tdata/source.png'
img2 = cv2.imread(img2)[:, :, ::-1].copy().transpose([2, 0, 1])
img2 = torch.cuda.FloatTensor(img2).unsqueeze(0)
img_cropped2 = torch.nn.functional.interpolate(img2,
                                               size=[640, 640],
                                               mode='bilinear')
# img_cropped2=(img_cropped2 - 127.5) / 128
loc, conf, landms = net(Variable(img_cropped2))

a = [[1, 8], [3, 4], [6, 5]]
print(np.argmax(a))

print(conf.shape)
print(round(np.argmax(conf.data.cpu().numpy()) / 2))
print(torch.argmax(conf))
print(torch.max(conf))
print(landms[0][16497])
print()
print()

cfg = cfg_re50

scale = torch.Tensor([img_cropped2.shape[1], img_cropped2.shape[0], img_cropped2.shape[1], img_cropped2.shape[0]])
scale = scale.to('cuda')
priorbox = PriorBox(cfg, image_size=(640, 640))
priors = priorbox.forward()
priors = priors.to('cuda')
prior_data = priors.data
boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
boxes = boxes * scale / 1
boxes = boxes.cpu().numpy()
scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
print(scores)
print(scores.shape)
print(np.argmax(scores))
print(np.max(scores))
print()
print()
landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
print(landms[16497])

scale1 = torch.Tensor([img_cropped2.shape[3], img_cropped2.shape[2], img_cropped2.shape[3], img_cropped2.shape[2],
                       img_cropped2.shape[3], img_cropped2.shape[2], img_cropped2.shape[3], img_cropped2.shape[2],
                       img_cropped2.shape[3], img_cropped2.shape[2]])
scale1 = scale1.to('cuda')
landms = landms * scale1 / 1
landms = landms.cpu().numpy()
# ignore low scores
inds = np.where(scores > 0.2)[0]
boxes = boxes[inds]
landms = landms[inds]
scores = scores[inds]
# keep top-K before NMS
order = scores.argsort()[::-1][:5000]
boxes = boxes[order]
landms = landms[order]
scores = scores[order]
# do NMS
dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
keep = py_cpu_nms(dets, 0.4)
# keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
dets = dets[keep, :]
landms = landms[keep]

# keep top-K faster NMS
dets = dets[:750, :]

landms = landms[:750, :]

dets = np.concatenate((dets, landms), axis=1)

# print(dets)
b = None
for a in dets:
    b = list(map(int, a))
print(img_cropped2.shape)

# # im=Image.fromarray(np.clip(a.transpose([1,2,0]),0,255).astype(np.uint8))
# # im.show()
# img_cropped2 = img_cropped2.Tdata.cpu().numpy()[0].astype(np.uint8)
#
# cv2.circle(img_cropped2, (b[5], b[6]), 1, (0, 0, 255), 4)
#
# cv2.circle(img_cropped2, (b[7], b[8]), 1, (0, 255, 255), 4)
#
# cv2.circle(img_cropped2, (b[9], b[10]), 1, (255, 0, 255), 4)
#
# cv2.circle(img_cropped2, (b[11], b[12]), 1, (0, 255, 0), 4)
#
# cv2.circle(img_cropped2, (b[13], b[14]), 1, (255, 0, 0), 4)
# name = "img/test.jpg"
# im=Image.fromarray(img_cropped2.transpose([1,2,0]))
# im.save(name)
# im.show()
#
# # cv2.imwrite(name, img_cropped2)

print(b)
