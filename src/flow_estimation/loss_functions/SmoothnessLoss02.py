# Shree KRISHNAya Namaha
# Edge-aware SmoothnessLoss in every MPI plane
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from pathlib import Path

import torch
import torch.nn.functional as F

from flow_estimation.loss_functions.LossFunctionParent01 import LossFunctionParent

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class SmoothnessLoss(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.scale_weights = loss_configs['scale_weights']
        self.flow_weight_coefficient = loss_configs['flow_weight_coefficient']
        self.bidirectional_flow = self.configs['flow_estimation']['frame_predictor']['bidirectional_flow']
        return

    def compute_loss(self, input_dict: dict, output_dict: dict):
        frame1 = input_dict['frame1']
        frame2 = input_dict['frame2']
        mask1 = input_dict['mask1']
        mask2 = input_dict['mask2']
        mpi1_alpha = input_dict['mpi1_alpha']
        mpi2_alpha = input_dict['mpi2_alpha']
        flows12 = output_dict['estimated_mpi_flows12']
        if self.bidirectional_flow:
            flows21 = output_dict['estimated_mpi_flows21']

        num_levels = len(flows12)
        sm_list = []
        for i in range(num_levels):
            mpi_flow12, mpi_flow12_mask = flows12[i]
            if self.bidirectional_flow:
                mpi_flow21, mpi_flow21_mask = flows21[i]

            b, _, h, w, d = mpi_flow12.shape
            frame1_scaled = F.interpolate(frame1, (h, w), mode='area')
            frame2_scaled = F.interpolate(frame2, (h, w), mode='area')
            mask1_scaled = F.interpolate(mask1, (h, w), mode='area')
            mask2_scaled = F.interpolate(mask2, (h, w), mode='area')
            mpi1_alpha_scaled = F.interpolate(mpi1_alpha, (h, w, d), mode='area')
            mpi2_alpha_scaled = F.interpolate(mpi2_alpha, (h, w, d), mode='area')
            sm1 = self.compute_smoothness_loss(mpi_flow12, mpi_flow12_mask, mpi1_alpha_scaled, frame1_scaled, mask1_scaled)
            if self.bidirectional_flow:
                sm2 = self.compute_smoothness_loss(mpi_flow21, mpi_flow21_mask, mpi2_alpha_scaled, frame2_scaled, mask2_scaled)
                scale_sm = (sm1 + sm2) / 2
            else:
                scale_sm = sm1
            sm_list.append(scale_sm)
        total_sm_loss = sum([w * l for w, l in zip(self.scale_weights, sm_list)])
        return total_sm_loss

    def compute_smoothness_loss(self, mpi_flow, mpi_flow_mask, mpi_alpha, frame, frame_mask, return_weights: bool = False):
        # All below checks verified
        # assert (mpi_alpha * (1 - mpi_flow_mask)).max() == 0

        flow_dy, flow_dx = self.compute_gradients(mpi_flow)
        alpha_dy, alpha_dx = self.compute_gradients(mpi_alpha)
        frame_dy, frame_dx = self.compute_gradients(frame)
        # Wherever alpha is zero, even if flow is non-zero, we can't impose smoothness
        weights_x = (1 - alpha_dx) * torch.exp(-torch.mean(torch.abs(frame_dx[..., None]), 1, keepdim=True) * self.flow_weight_coefficient)
        weights_y = (1 - alpha_dy) * torch.exp(-torch.mean(torch.abs(frame_dy[..., None]), 1, keepdim=True) * self.flow_weight_coefficient)

        loss_x = weights_x * flow_dx.abs() / 2
        loss_y = weights_y * flow_dy.abs() / 2
        sm_loss = loss_x.mean() / 2 + loss_y.mean() / 2
        if return_weights:
            return sm_loss, weights_x, weights_y
        return sm_loss

    @staticmethod
    def compute_gradients(image):
        grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
        grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
        return grad_y, grad_x
