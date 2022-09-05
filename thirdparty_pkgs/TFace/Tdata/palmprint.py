
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PolyUPalmprint(Dataset):
    def __init__(self,
                 root_dir,
                 local_rank,
                 transform=transforms.Compose([
                     transforms.Resize(112),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(0.5, 0.5),
                 ]),
                 train=True,
                 train_split=0.8
                 ):
        super().__init__()
        self.image_path = root_dir
        self.local_rank = local_rank

        all_img_list = []
        for i in range(1, 501):
            # 500 people
            for j in range(1, 3):
                # left and right hand
                for k in range(1, 7):
                    # 6 captures
                    label = i*3-3+j-1
                    all_img_list.append(
                        ("%03d/%d_%02d_s.bmp" % (i, j, k), label)
                    )
        if train:
            self.img_list = all_img_list[:int(train_split*len(all_img_list))]
        else:
            self.img_list = all_img_list[int(train_split*len(all_img_list)):]
        self.transform = transform

    def __getitem__(self, index):
        cur_im = []
        for channel in ['R', 'G', 'B',
                        #  'I'
                        ]:
            im_path = os.path.join(self.image_path, "Multispectral_" + channel, self.img_list[index][0])
            x = Image.open(im_path)
            cur_im.append(self.transform(x))
        cur_im = torch.cat(cur_im, 0)
        return cur_im, self.img_list[index][1]

    def __len__(self, ):
        return len(self.img_list)


# class PolyUPalmprintGrayScale(Dataset):
#     def __init__(self,
#                  root_dir,
#                  local_rank,
#                  transform=transforms.Compose([
#                      transforms.Resize(112),
#                      transforms.RandomHorizontalFlip(),
#                      transforms.ToTensor(),
#                      transforms.Normalize(0.5, 0.5),
#                  ]),
#                  train=True,
#                  train_split=0.8
#                  ):
#         super().__init__()
#         self.image_path = root_dir
#         self.local_rank = local_rank
#
#         all_img_list = []
#         for i in range(1, 501):
#             # 500 people
#             for j in range(1, 3):
#                 # left and right hand
#                 for k in range(1, 7):
#                     # 6 captures
#                     label = i*3-3+j-1
#                     all_img_list.append(
#                         ("%03d/%d_%02d_s.bmp" % (i, j, k), label)
#                     )
#         if train:
#             self.img_list = all_img_list[:int(train_split*len(all_img_list))]
#         else:
#             self.img_list = all_img_list[int(train_split*len(all_img_list)):]
#         self.transform = transform
#
#     def __getitem__(self, index):
#         cur_im = []
#         for channel in ['R', 'G', 'B',
#                         #  'I'
#                         ]:
#             im_path = os.path.join(self.image_path, "Multispectral_" + channel, self.img_list[index][0])
#             x = Image.open(im_path)
#             cur_im.append(self.transform(x))
#         cur_im = cur_im[0]*0.299 + cur_im[1]*0.587 + cur_im[2]*0.114  # convert to gray
#         return cur_im, self.img_list[index][1]
#
#     def __len__(self, ):
#         return len(self.img_list)


class TongJiPalmprint(Dataset):
    def __init__(self,
                 root_dir,
                 local_rank,
                 transform=transforms.Compose([
                     transforms.Resize(112),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(0.5, 0.5),
                 ]),
                 train=True,
                 train_split=0.8
                 ):
        super().__init__()
        self.image_path = root_dir
        self.local_rank = local_rank

        all_img_list = []
        for session in range(2):
            for personid in range(300):
                for idx in range(20):
                    label = session*300+personid
                    all_img_list.append(
                        ('session%d/%05d.bmp'%(session+1, personid*20+idx+1), label)
                    )
        if train:
            self.img_list = all_img_list[:int(train_split*len(all_img_list))]
        else:
            self.img_list = all_img_list[int(train_split*len(all_img_list)):]
        self.transform = transform

    def __getitem__(self, index):
        im_path = os.path.join(self.image_path, self.img_list[index][0])
        x = Image.open(im_path)
        cur_im = self.transform(x).repeat([3, 1, 1])  # convert to RGB like
        return cur_im, self.img_list[index][1]

    def __len__(self, ):
        return len(self.img_list)
