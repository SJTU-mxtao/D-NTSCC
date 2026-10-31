from conditional_entropy_model import ConditionalEntropyBottleneck, lower_bound
from WITT_modules import *
from Swin_Encoder import SwinEncoder
from Swin_Decoder import SwinDecoder
from stn_net import SpatialTransform
import entropy_model


class HyperAnalysisTransform(nn.Module):
    def __init__(self, num_filters=192):
        super(HyperAnalysisTransform, self).__init__()
        self.conv_h1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.relu_h1 = nn.ReLU()
        self.conv_h2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.relu_h2 = nn.ReLU()
        self.conv_h3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2, bias=False)

    def forward(self, x):
        x = self.conv_h1(x)
        x = self.relu_h1(x)
        x = self.conv_h2(x)
        x = self.relu_h2(x)
        x = self.conv_h3(x)

        return x


class HyperSynthesisTransform(nn.Module):
    def __init__(self, num_filters=192, num_filters_out=192):
        super(HyperSynthesisTransform, self).__init__()
        self.conv_h4 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h4 = nn.ReLU()
        self.conv_h5 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h5 = nn.ReLU()
        self.conv_h6 = nn.ConvTranspose2d(num_filters, num_filters_out * 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_h4(x)
        x = self.relu_h4(x)
        x = self.conv_h5(x)
        x = self.relu_h5(x)
        x = self.conv_h6(x)
        miu = x[:, :int(x.shape[1]/2), :, :]
        sigma = x[:, int(x.shape[1] / 2):, :, :]

        return miu, sigma


class Upsample(nn.Module):
    def __init__(self, num_filters=192, num_filters_out=192):
        super(Upsample, self).__init__()
        self.conv_h4 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h4 = nn.ReLU()
        self.conv_h5 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.relu_h5 = nn.ReLU()
        self.conv_h6 = nn.ConvTranspose2d(num_filters, num_filters_out, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_h4(x)
        x = self.relu_h4(x)
        x = self.conv_h5(x)
        x = self.relu_h5(x)
        x = self.conv_h6(x)

        return x


class FEncoder(nn.Module):
    def __init__(self, config):
        super(FEncoder, self).__init__()

        self.config = config

        self.encoder_kwargs = dict(
                img_size=(128, 256), patch_size=2, in_chans=3,
                embed_dims=[128, 160, 192, 256], depths=[2, 2, 6, 2], num_heads=[4, 8, 8, 8], C=256,  # the parameter C does not effect the architecture
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        self.encoder = SwinEncoder(**self.encoder_kwargs)
        self.encoder = SwinEncoder(**self.encoder_kwargs)

        self.ha1 = HyperAnalysisTransform(256)
        self.hs1 = HyperSynthesisTransform(256, 256)

        self.ha2 = HyperAnalysisTransform(256)
        self.hs2 = HyperSynthesisTransform(256, 256)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(256)

        self.conditional_entropy_bottleneck = ConditionalEntropyBottleneck()
        self.bound = 0.05

    def forward(self, x1, x2):

        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        y1_re = y1.reshape(-1, 8, 16, 256).permute(0, 3, 1, 2)
        y2_re = y2.reshape(-1, 8, 16, 256).permute(0, 3, 1, 2)

        z1 = self.ha1(y1_re)
        z2 = self.ha2(y2_re)

        z1_tilde, z2_tilde, z_likelihoods, z1_predicted, z2_predicted = self.entropy_bottleneck(z1, z2)

        miu_1, sigma_1 = self.hs1(z1_tilde)
        miu_2, sigma_2 = self.hs1(z2_tilde)

        miu_1_predicted, sigma_1_predicted = self.hs1(z1_predicted)
        miu_2_predicted, sigma_2_predicted = self.hs1(z2_predicted)

        sigma_1 = sigma_1.reshape(-1, 256, 128).permute(0, 2, 1)
        sigma_2 = sigma_2.reshape(-1, 256, 128).permute(0, 2, 1)

        miu_1 = miu_1.reshape(-1, 256, 128).permute(0, 2, 1)
        miu_2 = miu_2.reshape(-1, 256, 128).permute(0, 2, 1)

        sigma_1_predicted = sigma_1_predicted.reshape(-1, 256, 128).permute(0, 2, 1)
        sigma_2_predicted = sigma_2_predicted.reshape(-1, 256, 128).permute(0, 2, 1)

        miu_1_predicted = miu_1_predicted.reshape(-1, 256, 128).permute(0, 2, 1)
        miu_2_predicted = miu_2_predicted.reshape(-1, 256, 128).permute(0, 2, 1)

        sigma_1 = lower_bound(sigma_1, self.bound)
        y1_tilde, y1_likelihoods = self.conditional_entropy_bottleneck(y1, sigma_1, miu_1)
        sigma_2 = lower_bound(sigma_2, self.bound)
        y2_tilde, y2_likelihoods = self.conditional_entropy_bottleneck(y2, sigma_2, miu_2)

        sigma_1_predicted = lower_bound(sigma_1_predicted, self.bound)
        _, y1_likelihoods_predicted = self.conditional_entropy_bottleneck(miu_1_predicted, sigma_1_predicted, miu_1_predicted)
        sigma_2_predicted = lower_bound(sigma_2_predicted, self.bound)
        _, y2_likelihoods_predicted = self.conditional_entropy_bottleneck(miu_2_predicted, sigma_2_predicted, miu_2_predicted)

        return y1_tilde, y2_tilde, y1_likelihoods, y2_likelihoods, y1_likelihoods_predicted, y2_likelihoods_predicted, z_likelihoods, \
               z1_tilde, z2_tilde, z1_predicted, z2_predicted



class FDecoder(nn.Module):
    def __init__(self, config):
        super(FDecoder, self).__init__()

        self.decoder_kwargs = dict(
                img_size=(128, 256),  # (256, 256)
                embed_dims=[512, 384, 320, 256], depths=[2, 6, 2, 2], num_heads=[8, 8, 8, 4], C=512,  # embed[0]=500, c=680
                window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                norm_layer=nn.LayerNorm, patch_norm=True,
            )
        self.decoder = SwinDecoder(**self.decoder_kwargs)

    def forward(self, x1, x2):
        x1, x2 = self.decoder(x1, x2)
        # x2 = self.decoder(x2)
        return x1, x2


class D_NTSC(nn.Module):
    def __init__(self, device, config):
        super(D_NTSC, self).__init__()

        self.device = device
        self.config = config

        self.encoder = FEncoder(self.config)
        self.decoder = FDecoder(self.config)

        self.spatial_transform = SpatialTransform(256)

    def token_channel_shuffle(self, x, groups):
        batch_size, seq_len, num_channels = x.size()
        channels_per_group = num_channels // groups

        x = x.view(batch_size, seq_len, groups, channels_per_group)

        x = torch.transpose(x, 2, 3).contiguous()

        x = x.view(batch_size, seq_len, num_channels)
        return x

    def forward(self, x1, x2):
        y1_tilde, y2_tilde, y1_likelihoods, y2_likelihoods, y1_likelihoods_predicted, y2_likelihoods_predicted, \
        z_likelihoods, z_1, z_2, z1_predicted, z2_predicted = self.encoder(x1, x2)

        y2_si, y1_si, theta_1, theta_2 = self.spatial_transform(y1_tilde.reshape(-1, 8, 16, 256).permute(0, 3, 1, 2), y2_tilde.reshape(-1, 8, 16, 256).permute(0, 3, 1, 2))

        y2_si = y2_si.reshape(y2_si.shape[0], y2_si.shape[1], -1).permute(0, 2, 1)
        y1_si = y1_si.reshape(y1_si.shape[0], y1_si.shape[1], -1).permute(0, 2, 1)

        input_1 = torch.concat([y1_tilde, y1_si], dim=-1)
        input_2 = torch.concat([y2_tilde, y2_si], dim=-1)

        x1_hat, x2_hat = self.decoder(input_1, input_2)

        return x1_hat, x2_hat, y1_likelihoods, y2_likelihoods, z_likelihoods, z_1, z_2, z1_predicted, z2_predicted
