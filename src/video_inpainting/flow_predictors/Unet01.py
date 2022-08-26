# Shree KRISHNAya Namaha
# U-Net
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import torch
import torch.nn.functional as F


class FlowPredictor(torch.nn.Module):
    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs
        self.conv1 = torch.nn.Conv3d(in_channels=4, out_channels=32, kernel_size=7, stride=(1, 1, 1), padding=3)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=5, stride=(2, 2, 1), padding=2)
        self.conv3 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=(2, 2, 1), padding=1)
        self.conv4 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=(2, 2, 1), padding=1)
        self.conv5 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=(2, 2, 1), padding=1)
        self.up = torch.nn.Upsample(scale_factor=(2, 2, 1), mode='nearest')
        self.conv6 = torch.nn.Conv3d(in_channels=128 + 128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv3d(in_channels=128 + 128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv3d(in_channels=64 + 64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv3d(in_channels=32 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
        return

    def forward(self, input_batch):
        input_mpi = torch.cat([input_batch['mpi_rgb'], input_batch['mpi_alpha']], dim=1)
        x1 = self.conv1(input_mpi)  # 256x256, 32
        x1 = F.relu(x1)
        x2 = self.conv2(x1)  # 128x128, 64
        x2 = F.relu(x2)
        x3 = self.conv3(x2)  # 64x64, 128
        x3 = F.relu(x3)
        x4 = self.conv4(x3)  # 32x32, 128
        x4 = F.relu(x4)
        x5 = self.conv5(x4)  # 32x32, 128
        x5 = F.relu(x5)

        x6 = self.up(x5)  # 32x32, 128
        x6 = torch.cat([x6, x4], dim=1)  # 32x32, 256+256
        x6 = self.conv6(x6)  # 32x32, 128
        x6 = F.relu(x6)
        x7 = self.up(x6)  # 64x64, 128
        x7 = torch.cat([x7, x3], dim=1)  # 64x64, 128+128
        x7 = self.conv7(x7)  # 64x64, 64
        x7 = F.relu(x7)
        x8 = self.up(x7)  # 128x128, 64
        x8 = torch.cat([x8, x2], dim=1)  # 128x128, 64+64
        x8 = self.conv8(x8)  # 128x128, 32
        x8 = F.relu(x8)
        x9 = self.up(x8)  # 256x256, 32
        x9 = torch.cat([x9, x1], dim=1)  # 256x256, 32+32
        x9 = self.conv9(x9)  # 256x256, 3
        disoccluded_flow = x9 * (1 - input_batch['mask'][:, :, :, :, None])
        result_dict = {
            'disoccluded_flow': disoccluded_flow,
        }
        return result_dict
