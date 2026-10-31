import argparse
from torch.utils.data import DataLoader
import torch
import os

from cityscapes import PairCityscape
from kitti import PairKitti
from train import train, train_sc
from network import D_NTSCC
from network_sc import D_NTSC
from utils import init_seeds

def mischandler(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


def main(config):
    init_seeds(1)

    if config.dataset == "kitti":
        train_set = PairKitti('./KITTI', 'train')
        test_set = PairKitti('./KITTI', 'test')
    elif config.dataset == "cityscapes":
        train_set = PairCityscape("./Cityscapes", 'train')
        test_set = PairCityscape("./Cityscapes", 'test')

    train_loader = DataLoader(train_set, config.batch_size, True, num_workers=1)
    test_loader = DataLoader(test_set, config.batch_size, True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.method == 'd-ntscc':
        net = D_NTSCC(device, config).to(device)

        train(net, config, train_loader, test_loader, device)

    elif config.method == 'd-ntsc':
        net = D_NTSC(device, config).to(device)

        train_sc(net, config, train_loader, test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=230)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--snr_train_1', type=float, default=5)
    parser.add_argument('--snr_train_2', type=float, default=5)
    parser.add_argument('--method', type=str, default='d-ntscc')  # d-ntscc or d-ntsc
    parser.add_argument('--a', type=float, default=0.07)
    parser.add_argument('--channel_mode', type=str, default='awgn')  # awgn or fading
    parser.add_argument('--loss', type=str, default='psnr')  # psnr or ssim
    parser.add_argument('--dataset', type=str, default='kitti')  # kitti, cityscapes
    parser.add_argument('--model_path', type=str, default='./models/models_kitti/d_ntscc')
    parser.add_argument('--result_path', type=str, default='./results/results_kitti/d_ntscc')
    parser.add_argument('--config_path', type=str, default='./cons_fig/dis')
    parser.add_argument('--data_path', type=str, default='./dataset')

    config = parser.parse_args()

    mischandler(config)
    main(config)

