import torch
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader as DataLoader, Dataset
from matplotlib import pyplot as plt
import numpy as np
import os
import random


class Cifar10(Dataset):
    def __init__(self, cifar10, transform_1, transform_2):
        self.cifar10 = cifar10
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.labels = self.cifar10.targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.cifar10[index][0]
        data_1 = self.transform_1(data)
        date_2 = self.transform_2(data)
        return data_1, date_2

"""lower correlation"""
# class Cifar10(Dataset):
#     def __init__(self, cifar10, transform_1, transform_2):
#         self.cifar10 = cifar10
#         self.transform_1 = transform_1
#         self.transform_2 = transform_2
#         self.labels = self.cifar10.targets
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         data = self.cifar10[index][0]
#         data_1 = self.transform_1(data)[:, :24, :24]
#         data_1 = transforms.Resize(32)(data_1)
#         data_2 = self.transform_2(data)[:, 8:, 8:]
#         data_2 = transforms.Resize(32)(data_2)
#         return data_1, data_2

"""no correlation"""
# class Cifar10(Dataset):
#     def __init__(self, cifar10, transform_1, transform_2):
#         self.cifar10 = cifar10
#         self.transform_1 = transform_1
#         self.transform_2 = transform_2
#         self.labels = self.cifar10.targets
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         data = self.cifar10[index][0]
#         index_ran = random.randrange(0, len(self.labels))
#         date_ran =self.cifar10[index_ran][0]
#         data_1 = self.transform_1(data)
#         data_2 = self.transform_2(date_ran)
#         return data_1, data_2


def data_loader(data_root, batch_size, transform_1, transform_2):

    transform_1 = transforms.Compose(transform_1)
    transform_2 = transforms.Compose(transform_2)

    cifar10_training = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=None,
        download=True,
    )

    train_set = Cifar10(cifar10_training, transform_1, transform_2)
    train_loader = DataLoader(train_set, batch_size, True)

    cifar10_testing = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=None,
        download=True,
    )
    test_set = Cifar10(cifar10_testing, transform_1, transform_2)
    test_loader = DataLoader(test_set, batch_size, True)

    return train_loader, test_loader
