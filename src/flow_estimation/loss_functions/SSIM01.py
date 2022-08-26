# Shree KRISHNAya Namaha
# Structural similarity loss
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from pathlib import Path

import torch
import torch.nn.functional as F

from flow_estimation.loss_functions.LossFunctionParent01 import LossFunctionParent
from flow_estimation.utils.WarperPytorch import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class SSIM(LossFunctionParent):
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
        ssim_list = []
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

            ssim1 = self.compute_ssim(mpi1_rgb_scaled, mpi1_alpha_scaled, mpi1_rgb_recon, mpi1_alpha_recon, occlusion_mask1)
            if self.bidirectional_flow:
                ssim2 = self.compute_ssim(mpi2_rgb_scaled, mpi1_alpha_scaled, mpi2_rgb_recon, mpi2_alpha_recon, occlusion_mask2)
                scale_ssim = (ssim1 + ssim2) / 2
            else:
                scale_ssim = ssim1
            ssim_list.append(scale_ssim)
        total_ssim = sum([w * l for w, l in zip(self.scale_weights, ssim_list)])
        return total_ssim

    @staticmethod
    def compute_ssim(mpi1_rgb, mpi1_alpha, mpi2_rgb_recon, mpi2_alpha_recon, occlusion_mask, md=1, return_map: bool = False):
        """
        Computes SSIM loss as (1 - SSIM) / 2
        :param mpi1_rgb:
        :param mpi1_alpha:
        :param mpi2_rgb_recon:
        :param mpi2_alpha_recon:
        :param occlusion_mask:
        :param md:
        :return:
        """
        patch_size = (2 * md + 1, 2 * md + 1, 1)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Wherever mpi2 is invalid, those regions are included in occlusion mask
        mask = mpi1_alpha * (1 - occlusion_mask)
        x = mpi1_rgb * mask
        y = mpi2_rgb_recon * mask

        mu_x = F.avg_pool3d(x, patch_size, 1, 0)
        mu_y = F.avg_pool3d(y, patch_size, 1, 0)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = F.avg_pool3d(x * x, patch_size, 1, 0) - mu_x_sq
        sigma_y = F.avg_pool3d(y * y, patch_size, 1, 0) - mu_y_sq
        sigma_xy = F.avg_pool3d(x * y, patch_size, 1, 0) - mu_x_mu_y

        nr = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        dr = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        ssim = nr / dr
        ssim_dist = torch.clamp((1 - ssim) / 2, 0, 1)
        mean_ssim_dist = torch.mean(ssim_dist) / (torch.mean(mask) + 1e-12)
        if return_map:
            ssim_dist = F.pad(ssim_dist, (1, 1, 1, 1), mode='constant', value=0)
            return mean_ssim_dist, ssim_dist
        return mean_ssim_dist
