# Shree KRISHNAya Namaha
# Mean absolute error loss
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from pathlib import Path

import torch
import torch.nn.functional as F

from flow_estimation.loss_functions.LossFunctionParent01 import LossFunctionParent
from flow_estimation.utils.WarperPytorch import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class MAE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        self.warper = Warper(configs['device'])
        self.scale_weights = loss_configs['scale_weights']
        self.bidirectional_flow = self.configs['flow_estimation']['frame_predictor']['bidirectional_flow']
        return
    
    def compute_loss(self, input_dict: dict, output_dict: dict):
        mpi1_rgb = input_dict['mpi1_rgb']
        mpi2_rgb = input_dict['mpi2_rgb']
        mpi1_alpha = input_dict['mpi1_alpha']
        mpi2_alpha = input_dict['mpi2_alpha']
        flows12 = output_dict['estimated_mpi_flows12']
        if 'mpi1_occlusion_mask' in output_dict:
            high_res_occ_mask1 = output_dict['mpi1_occlusion_mask']
        else:
            high_res_occ_mask1 = torch.zeros_like(mpi1_alpha)
        if self.bidirectional_flow:
            flows21 = output_dict['estimated_mpi_flows21']
            if 'mpi2_occlusion_mask' in output_dict:
                high_res_occ_mask2 = output_dict['mpi2_occlusion_mask']
            else:
                high_res_occ_mask2 = torch.zeros_like(mpi2_alpha)

        num_levels = len(flows12)
        mae_list = []
        for i in range(num_levels):
            mpi_flow12, mpi_flow12_mask = flows12[i]
            if self.bidirectional_flow:
                mpi_flow21, mpi_flow21_mask = flows21[i]

            b, _, h, w, d = mpi_flow12.shape
            # Mask aware downsampling
            mpi1_rgb_scaled = F.interpolate(mpi1_rgb, (h, w, d), mode='area')
            mpi2_rgb_scaled = F.interpolate(mpi2_rgb, (h, w, d), mode='area')
            mpi1_alpha_scaled = F.interpolate(mpi1_alpha, (h, w, d), mode='area')
            mpi2_alpha_scaled = F.interpolate(mpi2_alpha, (h, w, d), mode='area')
            mpi1_rgb_scaled = mpi1_rgb_scaled / (mpi1_alpha_scaled + 1e-3)
            mpi2_rgb_scaled = mpi2_rgb_scaled / (mpi2_alpha_scaled + 1e-3)
            mpi1_alpha_scaled = (mpi1_alpha_scaled > 0.05).float()
            mpi2_alpha_scaled = (mpi2_alpha_scaled > 0.05).float()

            mpi1_rgb_recon, mpi1_alpha_recon = self.warper.bilinear_interpolation_mpi(
                mpi2_rgb_scaled, mpi2_alpha_scaled, mpi_flow12, mpi_flow12_mask, is_image=True)
            if self.bidirectional_flow:
                mpi2_rgb_recon, mpi2_alpha_recon = self.warper.bilinear_interpolation_mpi(
                    mpi1_rgb_scaled, mpi1_alpha_scaled, mpi_flow21, mpi_flow12_mask, is_image=True)

            if i == 0:
                occlusion_mask1 = high_res_occ_mask1
                if self.bidirectional_flow:
                    occlusion_mask2 = high_res_occ_mask2
            else:
                occlusion_mask1 = F.interpolate(high_res_occ_mask1, (h, w, d), mode='nearest')
                if self.bidirectional_flow:
                    occlusion_mask2 = F.interpolate(high_res_occ_mask2, (h, w, d), mode='nearest')

            mae1 = self.compute_mae(mpi1_rgb_scaled, mpi1_alpha_scaled, mpi1_rgb_recon, mpi1_alpha_recon, occlusion_mask1)
            if self.bidirectional_flow:
                mae2 = self.compute_mae(mpi2_rgb_scaled, mpi1_alpha_scaled, mpi2_rgb_recon, mpi2_alpha_recon, occlusion_mask2)
                scale_mae = (mae1 + mae2) / 2
            else:
                scale_mae = mae1
            mae_list.append(scale_mae)
        total_mae = sum([w * l for w, l in zip(self.scale_weights, mae_list)])
        return total_mae

    @staticmethod
    def compute_mae(mpi1_rgb, mpi1_alpha, mpi2_rgb_recon, mpi2_alpha_recon, occlusion_mask):
        # Wherever mpi2 is invalid, those regions are included in occlusion mask
        mask = mpi1_alpha * (1 - occlusion_mask)
        masked_error_rgb = mask * (mpi1_rgb - mpi2_rgb_recon)
        masked_error_alpha = mask * (mpi1_alpha - mpi2_alpha_recon)
        mae_rgb = torch.mean(torch.abs(masked_error_rgb)) / (torch.mean(mask) + 1e-12)
        mae_alpha = torch.mean(torch.abs(masked_error_alpha)) / (torch.mean(mask) + 1e-12)
        mae = mae_rgb + mae_alpha
        return mae
