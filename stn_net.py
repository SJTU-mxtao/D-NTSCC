from torch import nn
import torch
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class STN(nn.Module):
    def __init__(self, c_in):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.cnn1 = nn.Sequential(
            nn.Conv2d(c_in, 256, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2, stride=[1, 2]),
            nn.ReLU(True))
        # self.se1 = SELayer(32)
        self.cnn2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))
        # self.se2 = SELayer(32)
        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))
        # self.se3 = SELayer(32)
        self.cnn4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(2, stride=2),  # do not have this
            nn.ReLU(True))
        # self.cnn5 = nn.Sequential(   # do not have this
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True))

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 1 * 2, 128),  # 64 * 1 * 2 or 64 * 3 * 2
            nn.ReLU(True),
            nn.Linear(128, 8)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x_from, x_to):
        input_concat = torch.concat([x_from, x_to], dim=1)
        x = self.cnn1(input_concat)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        # x = self.cnn5(x)  # do not have this

        x = x.reshape(-1, 64 * 1 * 2)  # (64 * 1 * 2)

        theta_free = self.fc_loc(x)
        last_element = torch.tensor([[1]]).repeat_interleave(x_from.shape[0], dim=0).to(x_from.device)
        theta = torch.concat([theta_free, last_element], dim=1)

        theta = theta.view(-1, 3, 3)
        return theta


def custom_perspective_transform(image, perspective_matrix):
    n, c, h, w = image.shape
    device = image.device

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device)
    )
    ones = torch.ones_like(grid_x)

    grid_homogeneous = torch.stack([grid_x, grid_y, ones], dim=0).reshape(3, -1).unsqueeze(0).repeat_interleave(image.shape[0], dim=0)

    grid_transformed = torch.bmm(perspective_matrix, grid_homogeneous)  # [3, H*W]

    xy = (grid_transformed[:, :2, :] / (grid_transformed[:, 2:3, :] + 1e-8)).reshape(image.shape[0], 2, h, w).permute(0, 2, 3, 1)

    transformed_image = F.grid_sample(image, xy, align_corners=True)

    return transformed_image


class SpatialTransform(nn.Module):
    def __init__(self, c_in=512):
        super(SpatialTransform, self).__init__()

        self.transform_module = STN(c_in * 2)

    def forward(self, x, y):

        theta_yt = self.transform_module(y, x)
        theta_xt = self.transform_module(x, y)

        y_t = custom_perspective_transform(y, theta_yt)
        x_t = custom_perspective_transform(x, theta_xt)

        return x_t, y_t, theta_xt, theta_yt

