import torch
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse
from tqdm import tqdm


def PSNR(tensor_org, tensor_trans):
    # total_psnr = 0
    origin = ((tensor_org + 1) / 2).clamp(0.0, 1.0)
    trans = ((tensor_trans + 1) / 2).clamp(0.0, 1.0)
    # for i in range(np.size(trans, 0)):
    #     psnr = 0
    #     for j in range(np.size(trans, 1)):
    #         psnr_temp = comp_psnr(origin[i, j, :, :], trans[i, j, :, :])
    #         psnr = psnr + psnr_temp
    #     psnr /= 3
    #     total_psnr += psnr
    mse = torch.mean((origin - trans) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    total_psnr = psnr * tensor_org.shape[0]

    return total_psnr


def init_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_percentage(code, mod, epoch, channel_use, name, snr, tao, path):
    code = code.reshape(-1)
    index = [i for i in range(len(code))]
    random.shuffle(index)
    code = code[index]
    code = code.reshape(-1, 2).cpu()

    I_point = torch.unique(code.reshape(-1))

    if mod == '16qam':
        order = 16
    elif mod == '64qam':
        order = 64

    I, Q = torch.meshgrid(I_point, I_point)
    map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(I.shape[0] ** 2, 2)
    per_s = []
    fig = plt.figure(dpi=300)
    ax = Axes3D(fig)
    fig.add_axes(ax)
    for i in range(I.shape[0] ** 2):
        temp = torch.sum(torch.abs(code - map[i, :]), dim=1)
        num = code.shape[0] - torch.count_nonzero(temp).item()
        per = num / code.shape[0]
        per_s.append(per)
        # plt.plot(map[i, 0], map[i, 1], marker="o", markersize=per * 100, color='b')
    # plt.show()
    per_s = torch.tensor(per_s).cpu()
    height = np.zeros_like(per_s)
    width = depth = 0.05
    surf = ax.bar3d(I.ravel(), Q.ravel(), height, width, depth, per_s, zsort='average', color='palegreen', alpha=0.6)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    file_name = path + '/{}_{}_{}_{}/'.format(name, mod, channel_use, tao) + '{}'.format(snr)
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    fig.savefig(file_name + '/{}'.format(epoch))
    # plt.show()
    plt.close()

    fig = plt.figure(dpi=300)
    for k in range(I.shape[0] ** 2):
        plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='palegreen')
    fig.savefig(file_name + '/scatter_{}'.format(epoch))

    plt.close()
    # h_single = torch.mean(torch.sum(-code_prob * torch.log(code_prob + 1e-9), dim=-1))
    # print(h_single)

    # if mod == '64qam' or mod == '256qam' or mod == '1024qam':
    #     fig = plt.figure(dpi=300)
    #     for k in range(order):
    #         plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='b')
    #     fig.savefig(file_name + '/scatter_{}'.format(epoch))
    #     # plt.show()
    #     plt.close()
