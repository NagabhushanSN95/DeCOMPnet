# Shree KRISHNAya Namaha
# Pyramid feature extractor. Applies feature extractor on every plane of MPI
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import torch

from flow_estimation.libraries.partial_convolution.partialconv3d import PartialConv3d


class FeatureExtractor(torch.nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs
        self.conv1a = PartialConv3d(in_channels=3, out_channels=16, kernel_size=3, stride=(2, 2, 1), padding=1, return_mask=True, multi_channel=True)
        self.conv1b = PartialConv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv2a = PartialConv3d(in_channels=16, out_channels=32, kernel_size=3, stride=(2, 2, 1), padding=1, return_mask=True, multi_channel=True)
        self.conv2b = PartialConv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv3a = PartialConv3d(in_channels=32, out_channels=64, kernel_size=3, stride=(2, 2, 1), padding=1, return_mask=True, multi_channel=True)
        self.conv3b = PartialConv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv4a = PartialConv3d(in_channels=64, out_channels=96, kernel_size=3, stride=(2, 2, 1), padding=1, return_mask=True, multi_channel=True)
        self.conv4b = PartialConv3d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv5a = PartialConv3d(in_channels=96, out_channels=128, kernel_size=3, stride=(2, 2, 1), padding=1, return_mask=True, multi_channel=True)
        self.conv5b = PartialConv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.conv6a = PartialConv3d(in_channels=128, out_channels=192, kernel_size=3, stride=(2, 2, 1), padding=1, return_mask=True, multi_channel=True)
        self.conv6b = PartialConv3d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, return_mask=True, multi_channel=True)
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.num_feature_levels = 6
        return

    def forward(self, input_batch):
        rgb = input_batch['mpi_rgb']  # (b, 3, h, w, d)
        alpha = input_batch['mpi_alpha']  # (b, 1, h, w, d)
        mask = alpha.repeat([1, 3, 1, 1, 1])
        x1a, m1a = self.conv1a(rgb, mask)  # 128x128, 16
        x1a = self.leaky_relu(x1a)
        x1b, m1b = self.conv1b(x1a, m1a)
        x1b = self.leaky_relu(x1b)

        x2a, m2a = self.conv2a(x1b, m1b)  # 64x64, 32
        x2a = self.leaky_relu(x2a)
        x2b, m2b = self.conv2b(x2a, m2a)
        x2b = self.leaky_relu(x2b)

        x3a, m3a = self.conv3a(x2b, m2b)  # 32x32, 64
        x3a = self.leaky_relu(x3a)
        x3b, m3b = self.conv3b(x3a, m3a)
        x3b = self.leaky_relu(x3b)

        x4a, m4a = self.conv4a(x3b, m3b)  # 16x16, 96
        x4a = self.leaky_relu(x4a)
        x4b, m4b = self.conv4b(x4a, m4a)
        x4b = self.leaky_relu(x4b)

        x5a, m5a = self.conv5a(x4b, m4b)  # 8x8, 128
        x5a = self.leaky_relu(x5a)
        x5b, m5b = self.conv5b(x5a, m5a)
        x5b = self.leaky_relu(x5b)

        x6a, m6a = self.conv6a(x5b, m5b)  # 4x4, 192
        x6a = self.leaky_relu(x6a)
        x6b, m6b = self.conv6b(x6a, m6a)
        x6b = self.leaky_relu(x6b)

        # TODO: Return one channel masks, unless the values are different in each channel
        result_dict = {
            'mpi_features': [(x1b, m1b), (x2b, m2b), (x3b, m3b), (x4b, m4b), (x5b, m5b), (x6b, m6b)],
        }
        return result_dict
