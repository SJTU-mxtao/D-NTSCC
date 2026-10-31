import torch
from tqdm import tqdm
from utils import PSNR
from torch import nn
import pandas as pd
from ssim_functions import ssim, ms_ssim, SSIM, MS_SSIM
from entropy_model import lower_bound_fn
import numpy as np
lower_bound = lower_bound_fn.apply


def EVAL(model, data_loader, device, config, epoch):
    model.eval()
    d_total_1 = 0
    d_total_2 = 0
    bpp_total_1 = 0
    bpp_total_2 = 0
    bpp_total_z = 0
    rate_total_1 = 0
    rate_total_2 = 0
    total = 0

    for batch_idx, (x1, x2) in enumerate(tqdm(data_loader)):
        x2 = x1
        x1, x2 = x1.to(device), x2.to(device)
        total += x1.shape[0]

        with torch.no_grad():

            x1_hat, x2_hat, y1_likelihoods, y2_likelihoods, z_likelihoods,\
            rate_1, rate_2, z1_tilte, z2_tilte, z1_predict, z2_predict = model(x1, x2)

        bpp_y1, bpp_y2, bpp_z = get_bpp(x1_hat, y1_likelihoods, y2_likelihoods, z_likelihoods)
        if config.loss == 'psnr':
            d_1 = PSNR(x1, x1_hat)
            d_2 = PSNR(x2, x2_hat)
        elif config.loss == 'ssim':
            d_1 = ms_ssim((x1 + 1) / 2, (x1_hat.detach() + 1) / 2, data_range=1.0, win_size=7) * x1.shape[0]
            d_2 = ms_ssim((x2 + 1) / 2, (x2_hat.detach() + 1) / 2, data_range=1.0, win_size=7) * x1.shape[0]

        d_total_1 += d_1
        d_total_2 += d_2
        bpp_total_1 += bpp_y1
        bpp_total_2 += bpp_y2
        bpp_total_z += bpp_z
        rate_total_1 += rate_1
        rate_total_2 += rate_2

    d_total_1 /= total
    d_total_2 /= total
    bpp_total_1 /= total
    bpp_total_2 /= total
    bpp_total_z /= total
    rate_total_1 /= total
    rate_total_2 /= total

    return d_total_1, d_total_2, bpp_total_1, bpp_total_2, bpp_total_z, rate_total_1, rate_total_2


def EVAL_sc(model, data_loader, device, config, epoch):
    model.eval()
    d_total_1 = 0
    d_total_2 = 0
    bpp_total_1 = 0
    bpp_total_2 = 0
    bpp_total_z = 0
    total = 0

    for batch_idx, (x1, x2) in enumerate(tqdm(data_loader)):
        x1, x2 = x1.to(device), x2.to(device)
        total += x1.shape[0]

        with torch.no_grad():

            x1_hat, x2_hat, y1_likelihoods, y2_likelihoods, z_likelihoods, z_1, z_2, _, _ = model(x1, x2)

        bpp_y1, bpp_y2, bpp_z = get_bpp(x1_hat, y1_likelihoods, y2_likelihoods, z_likelihoods)
        if config.loss == 'psnr':
            d_1 = PSNR(x1, x1_hat)
            d_2 = PSNR(x2, x2_hat)
        elif config.loss == 'ssim':
            d_1 = ms_ssim((x1 + 1) / 2, (x1_hat.detach() + 1) / 2, data_range=1.0, win_size=7)
            d_2 = ms_ssim((x2 + 1) / 2, (x2_hat.detach() + 1) / 2, data_range=1.0, win_size=7)

        d_total_1 += d_1
        d_total_2 += d_2
        bpp_total_1 += bpp_y1
        bpp_total_2 += bpp_y2
        bpp_total_z += bpp_z

    d_total_1 /= total
    d_total_2 /= total
    bpp_total_1 /= total
    bpp_total_2 /= total
    bpp_total_z /= total

    return d_total_1, d_total_2, bpp_total_1, bpp_total_2, bpp_total_z


def get_bpp(x_recon, likelihoods_y1, likelihoods_y2, z_likelihoods):  # Returns calculated bpp for train and test
    size_est = (-np.log(2) * x_recon.numel() / 3)
    bpp_y1 = (torch.sum(torch.log(likelihoods_y1))) / size_est
    bpp_y2 = (torch.sum(torch.log(likelihoods_y2))) / size_est
    bpp_z = (torch.sum(torch.log(z_likelihoods))) / size_est
    return bpp_y1, bpp_y2, bpp_z


def train(net, config, train_loader, test_loader, device):

    bottleneck_params = net.encoder.entropy_bottleneck.parameters()
    other_params = [param for name, param in net.named_parameters() if
                    "encoder.entropy_bottleneck" not in name]

    optimizer = torch.optim.Adam([
        {'params': bottleneck_params, 'weight_decay': 5e-5},
        {'params': other_params, 'weight_decay': 5e-5}
    ], lr=config.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.train_iters, T_mult=1, eta_min=1e-6, last_epoch=-1)

    results = {'epoch': [], 'psnr_1': [], 'psnr_2': [], 'bpp_1': [], 'bpp_2': [], 'bpp_z': [], 'rate_1': [], 'rate_2': []}
    epochs = config.train_iters

    loss_mse = nn.MSELoss()

    for epoch in range(epochs):
        net.train()
        d_total_1 = 0
        d_total_2 = 0
        total = 0
        bpp_total_1_train = 0
        bpp_total_2_train = 0
        bpp_total_z_train = 0
        rate_1_total = 0
        rate_2_total = 0
        epoch_loss = []

        for i, (x1, x2) in enumerate(tqdm(train_loader)):
            total += x1.shape[0]
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()

            x1_hat, x2_hat, y1_likelihoods, y2_likelihoods, z_likelihoods, rate_1, rate_2, \
            z1_tilde, z2_tilde, z1_predicted, z2_predicted = net(x1, x2)
            bpp_y1, bpp_y2, bpp_z = get_bpp(x1_hat, y1_likelihoods, y2_likelihoods, z_likelihoods)
            if config.loss == 'psnr':
                loss_image = config.a * (255 ** 2) * (loss_mse(x1, x1_hat) + loss_mse(x2, x2_hat)) + (bpp_y1 + bpp_y2 + bpp_z)
            elif config.loss == 'ssim':
                loss_image = config.a * (255 ** 2) * (2 - ms_ssim((x1 + 1) / 2, (x1_hat + 1) / 2, data_range=1.0, win_size=7) \
                             - ms_ssim((x2 + 1) / 2, (x2_hat + 1) / 2, data_range=1.0, win_size=7)) + (bpp_y1 + bpp_y2 + bpp_z)
            loss = loss_image
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().item())

            bpp_total_1_train += bpp_y1
            bpp_total_2_train += bpp_y2
            bpp_total_z_train += bpp_z
            rate_1_total += rate_1
            rate_2_total += rate_2

            if config.loss == 'psnr':
                d1 = PSNR(x1, x1_hat.detach())
                d2 = PSNR(x2, x2_hat.detach())
            elif config.loss == 'ssim':
                d1 = ms_ssim((x1 + 1) / 2, (x1_hat.detach() + 1) / 2, data_range=1.0, win_size=7) * x1.shape[0]
                d2 = ms_ssim((x2 + 1) / 2, (x2_hat.detach() + 1) / 2, data_range=1.0, win_size=7) * x1.shape[0]
            d_total_1 += d1
            d_total_2 += d2

        bpp_y1_ave = bpp_total_1_train / total
        bpp_y2_ave = bpp_total_2_train / total
        bpp_z_ave = bpp_total_z_train / total
        rate_1_ave = rate_1_total / total
        rate_2_ave = rate_2_total / total
        loss_ave = sum(epoch_loss) / len(epoch_loss)

        scheduler.step()

        d_ave_1 = d_total_1 / total
        d_ave_2 = d_total_2 / total

        print('epoch: {:d}, loss: {:.6f}, lr: {:.6f}'.format(epoch, loss_ave, optimizer.state_dict()['param_groups'][0]['lr']))
        print('train distortion: user 1 {:.3f}, user 2 {:.3f}'.format(d_ave_1, d_ave_2))
        print('train ave bpp y1: {:.6f}, y2: {:.6f}, z: {:.6f}'.format(bpp_y1_ave, bpp_y2_ave, bpp_z_ave))
        print('train ave rate y1: {:.6f}, y2: {:.6f}'.format(rate_1_ave, rate_2_ave))

        if epoch % 10 == 0:
            d1_test, d2_test, bpp_total_1, bpp_total_2, bpp_total_z, rate_total_1_test, \
            rate_total_2_test = EVAL(net, test_loader, device, config, epoch)

            print('User 1: {:.3f}, User 2: {:.3f}'.format(d1_test, d2_test))
            print('test ave bpp y1: {:.6f}, y2: {:.6f}, z: {:.6f}'.format(bpp_total_1, bpp_total_2, bpp_total_z))
            print('test ave rate y1: {:.6f}, y2: {:.6f}'.format(rate_total_1_test, rate_total_2_test))

            results['epoch'].append(epoch)
            results['psnr_1'].append(d1_test.cpu().numpy())
            results['psnr_2'].append(d2_test.cpu().numpy())
            results['bpp_1'].append(bpp_total_1.cpu().numpy())
            results['bpp_2'].append(bpp_total_2.cpu().numpy())
            results['bpp_z'].append(bpp_total_z.cpu().numpy())
            results['rate_1'].append(rate_total_1_test.cpu().numpy())
            results['rate_2'].append(rate_total_2_test.cpu().numpy())

            model_name = '/d_ntscc_a_{}_snr_{}.pth.tar'.format(config.a, config.snr_train_1)
            torch.save(net.state_dict(), config.model_path + model_name)
            print("Save model!")

            result_name = '/results_d_ntscc_a_{}_snr_{}.csv'.format(config.a, config.snr_train_1)
            data = pd.DataFrame(results)
            data.to_csv(config.result_path + result_name, index=False, header=False)

    model_name = '/d_ntscc_a_{}_snr_{}.pth.tar'.format(config.a, config.snr_train_1)
    torch.save(net.state_dict(), config.model_path + model_name)
    print("Save model!")

    result_name = '/results_d_ntscc_a_{}_snr_{}.csv'.format(config.a, config.snr_train_1)
    data = pd.DataFrame(results)
    data.to_csv(config.result_path + result_name, index=False, header=False)


def train_sc(net, config, train_loader, test_loader, device):

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=5e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.train_iters, T_mult=1, eta_min=1e-6, last_epoch=-1)

    results = {'epoch': [], 'psnr_1': [], 'psnr_2': [], 'bpp_1': [], 'bpp_2': [], 'bpp_z': []}
    epochs = config.train_iters

    loss_mse = nn.MSELoss()

    for epoch in range(epochs):
        net.train()
        d_total_1 = 0
        d_total_2 = 0
        total = 0
        error = 0
        bpp_total_1_train = 0
        bpp_total_2_train = 0
        bpp_total_z_train = 0
        epoch_loss = []

        for i, (x1, x2) in enumerate(tqdm(train_loader)):
            total += x1.shape[0]
            x1, x2 = x1.to(device), x2.to(device)

            optimizer.zero_grad()

            x1_hat, x2_hat, y1_likelihoods, y2_likelihoods, z_likelihoods, z_1, z_2, z1_predicted, z2_predicted = net(x1, x2)

            bpp_y1, bpp_y2, bpp_z = get_bpp(x1_hat, y1_likelihoods, y2_likelihoods, z_likelihoods)
            if config.loss == 'psnr':
                loss_image = config.a * (255 ** 2) * (loss_mse(x1, x1_hat) + loss_mse(x2, x2_hat)) + (bpp_y1 + bpp_y2 + bpp_z)

            elif config.loss == 'ssim':
                loss_ssim = 2 - ms_ssim((x1 + 1) / 2, (x1_hat + 1) / 2, data_range=1.0, win_size=7) \
                             - ms_ssim((x2 + 1) / 2, (x2_hat + 1) / 2, data_range=1.0, win_size=7)

                loss_image = config.a * (255 ** 2) * loss_ssim + (bpp_y1 + bpp_y2 + bpp_z)

            loss = loss_image
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().item())
            bpp_total_1_train += bpp_y1
            bpp_total_2_train += bpp_y2
            bpp_total_z_train += bpp_z

            if config.loss == 'psnr':
                d1 = PSNR(x1, x1_hat.detach())
                d2 = PSNR(x2, x2_hat.detach())
            elif config.loss == 'ssim':
                d1 = ms_ssim((x1 + 1) / 2, (x1_hat.detach() + 1) / 2, data_range=1.0, win_size=7)
                d2 = ms_ssim((x2 + 1) / 2, (x2_hat.detach() + 1) / 2, data_range=1.0, win_size=7)
            d_total_1 += d1
            d_total_2 += d2

        p_error =error / total
        bpp_y1_ave = bpp_total_1_train / total
        bpp_y2_ave = bpp_total_2_train / total
        bpp_z_ave = bpp_total_z_train / total
        loss_ave = sum(epoch_loss) / len(epoch_loss)

        scheduler.step()

        d_ave_1 = d_total_1 / total
        d_ave_2 = d_total_2 / total

        print('epoch: {:d}, loss: {:.6f}, lr: {:.6f}'.format(epoch, loss_ave, optimizer.state_dict()['param_groups'][0]['lr']))
        print('train distortion: user 1 {:.3f}, user 2 {:.3f}'.format(d_ave_1, d_ave_2))
        print('train ave bpp y1: {:.6f}, y2: {:.6f}, z: {:.6f}'.format(bpp_y1_ave, bpp_y2_ave, bpp_z_ave))
        print('predict error: {:.4f}'.format(p_error))

        if epoch % 10 == 0:
            d1_test, d2_test, bpp_total_1, bpp_total_2, bpp_total_z = EVAL_sc(net, test_loader, device, config, epoch)

            print('User 1: {:.3f}, User 2: {:.3f}'.format(d1_test, d2_test))
            print('test ave bpp y1: {:.6f}, y2: {:.6f}, z: {:.6f}'.format(bpp_total_1, bpp_total_2, bpp_total_z))

            results['epoch'].append(epoch)
            results['psnr_1'].append(d1_test.cpu().numpy())
            results['psnr_2'].append(d2_test.cpu().numpy())
            results['bpp_1'].append(bpp_total_1.cpu().numpy())
            results['bpp_2'].append(bpp_total_2.cpu().numpy())
            results['bpp_z'].append(bpp_total_z.cpu().numpy())

            model_name = '/d_ntsc_a_{}.pth.tar'.format(config.a)
            torch.save(net.state_dict(), config.model_path + model_name)
            print("Save model!")

            result_name = '/results_d_ntsc_a_{}.csv'.format(config.a)
            data = pd.DataFrame(results)
            data.to_csv(config.result_path + result_name, index=False, header=False)

    model_name = '/d_ntsc_a_{}.pth.tar'.format(config.a)
    torch.save(net.state_dict(), config.model_path + model_name)
    print("Save model!")

    result_name = '/results_d_ntsc_a_{}.csv'.format(config.a)
    data = pd.DataFrame(results)
    data.to_csv(config.result_path + result_name, index=False, header=False)

