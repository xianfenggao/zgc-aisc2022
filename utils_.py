import argparse
import os
import random
import sys
import time
import warnings
import copy

import cv2
import matplotlib
import numpy as np
from sympy import im
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as trans
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from pprint import pprint

from yaml import load

sys.path.append('./')
# sys.setrecursionlimit(1000)
from utils.misc import clear_requires_grad
from utils.data_augmentation.input_transformation import gkern
from models.FaceDetector.RetinaFace import arcface_src


def make_grid(mask):
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if i % 3 != 0 and j % 3 != 0:
                mask[i, j] = 0


def make_mask_with_landmarks(mask_path='dataset/Face/zgc-aisc2022/mask/grid_mask-v2.png',
                             make_grid_=True):

    src = np.asarray(arcface_src, dtype=np.int)[:2]
    for radiu in range(112):
        empty = np.zeros([112, 112])
        for point in src:
            cv2.circle(empty, point, radiu, (1), -1)
        if make_grid_:
            make_grid(empty)
        if np.sum(empty) > 1254:
            break
    radiu = radiu - 1
    print(radiu)
    empty = np.zeros([112, 112])
    for point in src:
        cv2.circle(empty, point, radiu, (1), -1)
    if make_grid_:
        make_grid(empty)
    assert islegal_mask(empty)
    empty = 255 - empty*255
    Image.fromarray(empty.astype(np.uint8)).save(mask_path)


def process_mask(mask_path):
    im = np.array(cv2.imread(mask_path)).sum(-1)
    im = np.where(im != 0, 255, 0)
    cv2.imwrite(mask_path, im)
    islegal_mask(im)


def islegal_mask(mask):
    if np.sum(mask) > 1254:
        return False
    has_searched = np.zeros_like(mask)
    neighbors = [
        [-1, 0],
        [1, 0],
        [0, 1],
        [0, -1]
    ]

    def dfs(i, j):
        if not has_searched[i, j] and mask[i, j] == 1:
            has_searched[i, j] = 1
            for nei in neighbors:
                if 0 <= i+nei[0] < 112 and 0 <= j+nei[1] < 112:
                    dfs(i+nei[0], j+nei[1])

    count = 0
    for i in range(112):
        for j in range(112):
            if not has_searched[i, j] and mask[i, j] == 1:
                dfs(i, j)
                count += 1
            if count > 5:
                return False
    return True


class zgc_aiscPairs(Dataset):
    def __init__(self, data_path='dataset/Face/zgc-aisc2022/data', trans=trans.ToTensor(), eval_nums=3000):
        super(zgc_aiscPairs, self).__init__()
        self.data_path = data_path
        self.trans = trans
        self.eval_nums = eval_nums
        assert 1 <= self.eval_nums <= 3000

    def __getitem__(self, item):
        item += 1
        source_file = os.path.join(self.data_path, "%04d.png" % item)
        source_img = Image.open(source_file)

        target_file = os.path.join(self.data_path, "%04d_compare.png" % item)
        target_img = Image.open(target_file)

        if self.trans is not None:
            source_img = self.trans(source_img)
            target_img = self.trans(target_img)

        return source_img, target_img

    def __len__(self):
        return self.eval_nums


def gen_probmatrix(kernel_size=21, sigma=3):

    kernel = gkern(kernel_size, sigma)
    src = np.asarray(arcface_src, dtype=np.int)
    empty = np.zeros([112, 112])
    for point in src:
        empty[
            point[1]-kernel_size//2: point[1]+kernel_size//2+1,
            point[0]-kernel_size//2: point[0]+kernel_size//2+1,
        ] += copy.deepcopy(kernel)
    empty = empty / np.sum(np.abs(empty))

    # cv2.imwrite("test_probmatrix.png", (empty*255).astype('uint8'))
    return empty


def sample_by_prob(probmatrix, points_num):
    shape = probmatrix.shape
    prob = probmatrix.ravel()
    indexes = list(range(len(prob)))
    sampled = np.random.choice(indexes, points_num, replace=False, p=prob)

    res = np.zeros_like(prob)
    res[sampled] = 1
    res = res.reshape(shape)
    return res


def keep_top5_neighbors(mask):
    has_searched = np.zeros_like(mask)
    neighbors = [
        [-1, 0],
        [1, 0],
        [0, 1],
        [0, -1]
    ]

    def dfs(i, j, this_search):
        if not has_searched[i, j] and mask[i, j] == 1:
            has_searched[i, j] = 1
            this_search.add((i, j))
            for nei in neighbors:
                if 0 <= i+nei[0] < 112 and 0 <= j+nei[1] < 112:
                    dfs(i+nei[0], j+nei[1], this_search)

    count = 0
    all_areas = []
    for i in range(112):
        for j in range(112):
            if not has_searched[i, j] and mask[i, j] == 1:
                this_search = set()
                dfs(i, j, this_search)
                # print(this_search)
                count += 1
                all_areas.append(this_search)

    all_areas.sort(key=lambda x: len(x), reverse=True)
    topk_areas = all_areas[:5]

    all_points = set()
    for area in topk_areas:
        all_points = all_points | area
    all_points = list(all_points)
    if len(all_points) > 1250:
        # print("Toal points: ", len(all_points))
        random.shuffle(all_points)
        all_points = all_points[:1250]
    all_points = set(all_points)
    mask = np.zeros((112, 112))
    for i in range(112):
        for j in range(112):
            if (i, j) in all_points:
                mask[i, j] = 1
    if not islegal_mask(mask):
        # print(mask, mask.sum(), len(all_points))
        mask = keep_top5_neighbors(mask)
    return mask


def draw_a_circle(mask, radiu, center, x_y_divide=.5):
    h, w = mask.shape[:2]
    for i in range(h):
        for j in range(w):
            x = abs(x-center[0])
            y = abs(y-center[1])
            if (x/x_y_divide)**2+y**2 <= radiu**2:
                mask[i, j] = 1
    return mask


def draw_a_square(mask, h_, w_, center):
    h, w = mask.shape[:2]
    for i in range(h):
        for j in range(w):
            x = abs(x-center[0])
            y = abs(y-center[1])
            if x <= h_ and y <= w_:
                mask[i, j] = 1
    return mask


if __name__ == '__main__':
    pass
