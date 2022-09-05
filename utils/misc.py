# coding: utf-8
import logging
import os
import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


# sys.data_path.append('./')
# matplotlib.use('Agg')


def accuracy(output, target, topk=(1,), distributed=False):
    """ Computes the precision@k for the specified values of k
    """
    maxk = max(topk)

    if distributed:
        from torch import distributed as dist
        world_size = dist.get_world_size()
        all_output = [output.clone().detach() for _ in range(world_size)]
        dist.all_gather(all_output, output)
        all_target = [target.clone().detach() for _ in range(world_size)]
        dist.all_gather(all_target, target)

        output = torch.cat(all_output, 0)
        target = torch.cat(all_target, 0)

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))

    return res, len(target)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        # self.name = name
        # self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """ Timer for count duration
    """

    def __init__(self):
        self.start_time = time.time()

    def get_duration(self):
        duration = time.time() - self.start_time
        self.start_time = time.time()
        return duration


def separate_resnet_bn_paras(modules):
    """ sepeated bn params and wo-bn params
    """
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, param in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(param)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id,
                              all_parameters))

    return paras_only_bn, paras_wo_bn


def clip_by_l2(diff, max_l2):
    l2 = diff.norm(2)
    if l2 < 1e-8:
        return diff
    else:
        factor = min(max_l2 / l2, 1.)
        return diff * factor


def clip_by_l_inf(diff, max_l_inf):
    return torch.clamp(diff, -max_l_inf, max_l_inf)


class Logger:
    def __init__(self, path):
        self.path = path
        if path != '':
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(self.path, 'a') as f:
                f.write('\n\nNew Log at %s\n' % (self.timestamp()))

    def print(self, message):
        print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()

    def timestamp(self):
        return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def read_image(path, shape=(600, 600)):
    return np.array(Image.open(path).resize(shape), dtype=float)


def create_mask(path, image_shape=(600, 600)):
    mask = cv2.resize(cv2.imread(path), image_shape)
    assert mask is not None
    mask = np.where(mask < 128, 1, 0)
    return mask


def shift_matrix(matrix, direction):
    # matrix is HxWxC
    # (5, 10) means shift towards top(5)/left(10)
    dh, dw = direction
    matrix = np.pad(matrix, ((abs(dh), abs(dh)), (abs(dw), abs(dw)), (0, 0)), 'constant', constant_values=(0, 0))

    if dh > 0:
        matrix = matrix[2 * dh:, :, :]
    elif dh < 0:
        matrix = matrix[:2 * dh, :, :]
    if dw > 0:
        matrix = matrix[:, 2 * dw:, :]
    elif dw < 0:
        matrix = matrix[:, :2 * dw, :]
    return matrix


def timestamp():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def tiled_gradients(model, imgs, y_true, loss_fun, steps_per_octave=100, tile_size=299, input_transform=None,
                    y_gt=None):
    """

    :param model:
    :param imgs:
    :param y_true: first round predictions
    :param loss_fun:
    :param steps_per_octave:
    :param tile_size:
    :param input_transform:
    :param y_gt:
    :return:
    """

    gradients = torch.zeros_like(imgs).detach()
    for _ in range(steps_per_octave):
        shift = [i for i in np.random.randint(low=-tile_size, high=tile_size, size=(2,))]
        img_rolled = torch.roll(imgs, shift, (2, 3))
        img_rolled = img_rolled.detach()
        img_rolled.requires_grad = True

        xs = range(0, imgs.shape[2], tile_size)[:-1]
        if len(xs) == 0:
            xs = [0]
        ys = range(0, imgs.shape[3], tile_size)[:-1]
        if len(ys) == 0:
            ys = [0]

        inner_grad = torch.zeros_like(img_rolled).detach()
        for x in xs:
            for y in ys:
                img_tile = img_rolled[:, :, x:x + tile_size, y:y + tile_size]
                if input_transform is not None:
                    y_ = model(input_transform(img_tile))
                else:
                    y_ = model(img_tile)
                loss = loss_fun(y_, y_true)
                if y_gt is not None:
                    loss += loss_fun(y_, y_gt)
                loss.backward()
                inner_grad = inner_grad + img_rolled.grad
                img_rolled.grad = None

        inner_grad = torch.roll(inner_grad, [-i for i in shift], (2, 3))
        inner_grad = inner_grad.div(torch.std(inner_grad) + 1e-8)

    gradients += inner_grad / steps_per_octave

    return gradients


def manual_all_seeds(seed=1111):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def set_logging(level=logging.INFO, file=None):
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(level)

    formater = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formater)

    logger.addHandler(ch)

    if file is not None:
        # 使用FileHandler输出到文件
        fh = logging.FileHandler(file)
        fh.setLevel(level)
        fh.setFormatter(formater)
        logger.addHandler(fh)


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NPS) of an adversarial patch.

    """

    def __init__(self, patch_size, printability_file='data/non_printability/nps-my-硫酸纸-desktop.txt'):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_size),
                                               requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


def clear_requires_grad(modules):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def batch_forward(model, x, batch_size):
    batch_num = len(x) // batch_size + int(len(x) % batch_size != 0)
    res = []
    for i in range(batch_num):
        batch_im = x[batch_size * i:(i + 1) * batch_size].cuda()
        res.append(model(batch_im))
    return torch.cat(res)
