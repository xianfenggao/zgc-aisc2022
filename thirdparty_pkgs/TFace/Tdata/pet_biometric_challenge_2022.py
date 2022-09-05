import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PetBiometric(Dataset):
    def __init__(self,
                 root_dir,
                 local_rank,
                 transform=transforms.Compose([
                     transforms.Resize((112, 112)),
                     # transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(0.5, 0.5),
                 ]),
                 train=True
                 ):
        super(PetBiometric, self).__init__()
        if train:
            self.img_path = os.path.join(root_dir, 'train')
            self.csv = os.path.join(root_dir, 'train', 'train_data.csv')
        else:
            self.img_path = os.path.join(root_dir, 'validation')
            self.csv = os.path.join(root_dir, 'validation', 'valid_data.csv')
        self.transform = transform
        self.local_rank = local_rank

        df = pd.read_csv(self.csv)
        self.train = train
        if train:
            self.labels = [int(i) for i in df['dog ID']]
            self.img_list = df['nose print image']
        else:
            # self.labels = [int(i) for i in df['dog ID']]
            self.img_a_list = df['imageA']
            self.img_b_list = df['imageB']

    def __len__(self):
        return len(self.img_list) if hasattr(self, 'img_list') else len(self.img_a_list)

    def __getitem__(self, index):
        if self.train:
            img_path = os.path.join(self.img_path, 'images', self.img_list[index])
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, self.labels[index]
        else:
            img_path = os.path.join(self.img_path, 'images', self.img_a_list[index])
            imga = Image.open(img_path)
            if self.transform is not None:
                imga = self.transform(imga)

            img_path = os.path.join(self.img_path, 'images', self.img_b_list[index])
            imgb = Image.open(img_path)
            if self.transform is not None:
                imgb = self.transform(imgb)
            return imga, imgb

    def get_labels(self):
        return self.labels
