# Shree KRISHNAya Namaha
# Extended from FramePredictor01.py. Uses normal convolutions instead of partial convolutions
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from pathlib import Path

import torch

from flow_estimation.utils.WarperPytorch import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class FramePredictor(torch.nn.Module):
    def __init__(self, configs: dict, feature_extractor, flow_estimator):
        super().__init__()
        self.configs = configs
        self.feature_extractor = feature_extractor
        self.flow_estimator = flow_estimator
        self.warper = Warper(configs['device'])
        self.bidirectional_flow = self.configs['frame_predictor']['bidirectional_flow']
        self.occlusion_mask_needed = self.configs['frame_predictor']['occlusion_mask']
        return

    def forward(self, input_batch: dict):
        # Extract features
        feature_extractor_input = {
            'mpi_rgb': input_batch['mpi1_rgb'],
            'mpi_alpha': input_batch['mpi1_alpha'],
        }
        feature_extractor_output = self.feature_extractor(feature_extractor_input)
        mpi1_features = feature_extractor_output['mpi_features']
        feature_extractor_input = {
            'mpi_rgb': input_batch['mpi2_rgb'],
            'mpi_alpha': input_batch['mpi2_alpha'],
        }
        feature_extractor_output = self.feature_extractor(feature_extractor_input)
        mpi2_features = feature_extractor_output['mpi_features']

        # Estimate flow
        flow_estimator_input = {
            'mpi1_features': mpi1_features,
            'mpi2_features': mpi2_features,
            'mpi1_alpha': input_batch['mpi1_alpha'],
            'mpi2_alpha': input_batch['mpi2_alpha'],
        }
        flow_estimator_output = self.flow_estimator(flow_estimator_input)
        flows12 = flow_estimator_output['estimated_mpi_flows12']
        output_batch = {
            'estimated_mpi_flows12': flows12,
        }

        if self.bidirectional_flow:
            flow_estimator_input = {
                'mpi1_features': mpi2_features,
                'mpi2_features': mpi1_features,
                'mpi1_alpha': input_batch['mpi2_alpha'],
                'mpi2_alpha': input_batch['mpi1_alpha'],
            }
            flow_estimator_output = self.flow_estimator(flow_estimator_input)
            flows21 = flow_estimator_output['estimated_mpi_flows12']
            output_batch['estimated_mpi_flows21'] = flows21

        if not self.training:
            # Warp frame1 to frame2 using flow12
            mpi_flow12 = flows12[0].clone()
            # mpi_flow12[:, :2] *= -0.5
            mpi1_rgb = input_batch['mpi1_rgb']
            mpi1_alpha = input_batch['mpi1_alpha']
            mpi1_depth = input_batch['mpi1_depth']
            warped_mpi2_rgb, warped_mpi2_alpha = self.warper.bilinear_splatting_mpi(mpi1_rgb, mpi1_alpha, mpi1_depth, mpi_flow12, None)
            warped_mpi2_weights = self.alpha2weights(warped_mpi2_alpha)
            warped_frame2a = torch.sum(warped_mpi2_rgb * warped_mpi2_weights, dim=4)
            mask2 = (torch.sum(warped_mpi2_alpha, dim=4) >= 1).float()
            output_batch['predicted_frame2'] = warped_frame2a
            output_batch['mask2'] = mask2

        if self.occlusion_mask_needed:
            occlusion_outputs = self.compute_occlusion_masks(input_batch, output_batch)
            output_batch.update(occlusion_outputs)
        return output_batch

    def compute_occlusion_masks(self, input_batch: dict, output_batch: dict):
        """
        Computes occlusion mask using bidirectional flow
        :return: occlusion_masks: (b, 1, h, w, d): 1 if occluded, 0 if visible
        """
        mpi_flow12 = output_batch['estimated_mpi_flows12'][0]
        mpi1_alpha = input_batch['mpi1_alpha']
        mpi2_alpha = input_batch['mpi2_alpha']
        mpi1_depth = input_batch['mpi1_depth']
        mpi2_depth = input_batch['mpi2_depth']

        occlusion_mask1 = self.compute_occlusion_mask(mpi1_alpha, mpi1_depth, mpi_flow12)
        output_dict = {
            'mpi1_occlusion_mask': occlusion_mask1,
        }

        if self.bidirectional_flow:
            mpi_flow21 = output_batch['estimated_mpi_flows21'][0]
            occlusion_mask2 = self.compute_occlusion_mask(mpi2_alpha, mpi2_depth, mpi_flow21)
            output_dict['mpi2_occlusion_mask'] = occlusion_mask2

        return output_dict

    def compute_occlusion_mask(self, mpi1_alpha, mpi1_depth, mpi_flow12):
        warped_alpha2 = self.warper.bilinear_splatting_mpi(mpi1_alpha, None, mpi1_depth, mpi_flow12, None, is_image=False)[0]
        visibility2 = self.alpha2visibility(warped_alpha2)
        warped_occ_mask2 = 1 - visibility2  # occluded if not visible
        occlusion_mask1 = self.warper.bilinear_interpolation_mpi(warped_occ_mask2, None, mpi_flow12, None, is_image=False)[0]
        # If a significant portion of current point moves into occluded region
        occlusion_mask1 = (occlusion_mask1 >= 0.5).float()
        return occlusion_mask1

    @staticmethod
    def alpha2visibility(mpi_alpha):
        first_plane_vis = torch.ones((*mpi_alpha.shape[:-1], 1)).to(mpi_alpha)
        visibility = torch.cumprod(torch.cat([first_plane_vis, 1. - mpi_alpha + 1e-10], -1), -1)[..., :-1]
        return visibility

    @staticmethod
    def alpha2weights(mpi_alpha):
        weights = mpi_alpha * FramePredictor.alpha2visibility(mpi_alpha)
        return weights
