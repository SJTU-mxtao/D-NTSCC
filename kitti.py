from random import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
import random as r
import os
import random as pyrandom


class PairKitti(Dataset):

    def __init__(self, path, set_type, resize=(128, 256)):
        super(Dataset, self).__init__()
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.resize = resize

        self.ar = []
        idx_path = './data_paths/KITTI_stereo_' + set_type + '.txt'

        with open(idx_path) as f:
            content = f.readlines()

        for i in range(0, len(content), 2):
            left_id = content[i].strip()
            right_id = content[i + 1].strip()
            self.ar.append((path + '/' + left_id, path + '/' + right_id))

        if set_type == 'train':
            self.transform = self.train_deterministic_cropping
        elif set_type == 'test' or set_type == 'val':
            self.transform = self.test_val_deterministic_cropping

    def compute_entropy(self, img):
        img = TF.center_crop(img, (370, 740))
        img = TF.resize(img, self.resize)

        re_img = np.asarray(img)
        hist, _ = np.histogram(re_img, bins=256, range=[0, 256])
        prob = hist / (re_img.shape[0] * re_img.shape[1] * re_img.shape[2])
        entropy = -np.sum(prob * np.log2(prob + np.finfo(float).eps))
        # print('Image entropy:', entropy)
        return entropy

    def offset_center_crop(self, img, crop_size, offset_x, offset_y):
        _, h, w = TF.get_dimensions(img)

        crop_h, crop_w = crop_size

        center_x, center_y = w // 2, h // 2

        top = max(center_y - crop_h // 2 + offset_y, 0)
        left = max(center_x - crop_w // 2 + offset_x, 0)

        return TF.crop(img, top, left, crop_h, crop_w)

    def train_deterministic_cropping(self, img, side_img):

        offset_x = r.randint(-50, 50) # original: 50

        img = self.offset_center_crop(img, (370, 740), offset_x, 0)
        side_img = self.offset_center_crop(side_img, (370, 740), offset_x, 0)

        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        # Random Horizontal Flip
        if random() > 0.5:
            img = TF.hflip(img)
            side_img = TF.hflip(side_img)

        # Convert to Tensor
        img = transforms.ToTensor()(img)
        side_img = transforms.ToTensor()(side_img)

        # Normalize
        img = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(img)
        side_img = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(side_img)

        return img, side_img

    def test_val_deterministic_cropping(self, img, side_img):
        # Center Crop
        img = TF.center_crop(img, (370, 740))
        side_img = TF.center_crop(side_img, (370, 740))

        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        # Convert to Tensor
        img = transforms.ToTensor()(img)
        side_img = transforms.ToTensor()(side_img)

        img = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(img)
        side_img = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(side_img)

        return img, side_img

    def __getitem__(self, index):
        left_path, right_path = self.ar[index]

        img = Image.open(left_path)
        side_img = Image.open(right_path)
        image_pair = self.transform(img, side_img)

        # return image_pair[0], image_pair[1], (left_path, right_path), index
        return image_pair[0], image_pair[1]

    def __len__(self):
        return len(self.ar)

    def __str__(self):
        return 'KITTI_stereo'


if __name__ == '__main__':
    ds = PairKitti(path='./', set_type='train')
    ds = DataLoader(dataset=ds)
    for data in ds:
        img, cor_img, idx, _ = data
        print(img.shape, idx)
    print(len(ds))
