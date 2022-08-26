# Shree KRISHNAya Namaha
#
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from pathlib import Path

import torch

from utils import MpiUtils
from video_inpainting.utils.Warper import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class FramePredictor(torch.nn.Module):
    def __init__(self, configs: dict, flow_predictor):
        super().__init__()
        self.configs = configs
        self.flow_predictor = flow_predictor
        self.warper = Warper(configs['device'])
        self.num_iterations = self.configs['video_inpainting']['frame_predictor']['num_iterations']
        return

    def forward(self, input_batch: dict):
        mpi_rgb = input_batch['mpi_rgb']
        mpi_alpha = input_batch['mpi_alpha']

        if self.training:
            num_iterations = 1
        else:
            num_iterations = self.num_iterations

        for i in range(num_iterations):
            mask = MpiUtils.alpha2mask(mpi_alpha)
            infill_input_batch = {
                'mpi_rgb': mpi_rgb,
                'mpi_alpha': mpi_alpha,
                # 'mpi_depth': mpi_depth,
                'mask': mask,
            }
            infill_output_batch = self.forward_one_iter(infill_input_batch)
            mpi_rgb = infill_output_batch['infilled_mpi_rgb']
            mpi_alpha = infill_output_batch['infilled_mpi_alpha']
            disoccluded_flow = infill_output_batch['disoccluded_flow']

        pred_frame, pred_mask = MpiUtils.alpha_compositing(mpi_rgb, mpi_alpha)
        output_batch = {
            'disoccluded_flow': disoccluded_flow,
            'infilled_mpi_rgb': mpi_rgb,
            'infilled_mpi_alpha': mpi_alpha,
            'pred_frame': pred_frame,
            'pred_mask': pred_mask,
        }
        return output_batch

    def forward_one_iter(self, input_batch: dict):
        flow_predictor_output = self.flow_predictor(input_batch)
        
        warper_inputs = {
            'mpi_rgb': input_batch['mpi_rgb'],
            'mpi_alpha': input_batch['mpi_alpha'],
            # 'mpi_depth': input_batch['mpi_depth'],
            'disoccluded_flow': flow_predictor_output['disoccluded_flow'],
        }
        warper_outputs = self.warper.bilinear_interpolation_mpi_flow2d(warper_inputs)
        mask = input_batch['mask'][:, :, :, :, None]
        infilled_mpi_rgb = mask * input_batch['mpi_rgb'] + (1 - mask) * warper_outputs['warped_rgb']
        infilled_mpi_alpha = mask * input_batch['mpi_alpha'] + (1 - mask) * warper_outputs['warped_alpha']
        # infilled_mpi_depth = mask * input_batch['mpi_depth'] + (1 - mask) * warper_outputs['warped_depth']

        output_batch = {
            'disoccluded_flow': flow_predictor_output['disoccluded_flow'],
            'infilled_mpi_rgb': infilled_mpi_rgb,
            'infilled_mpi_alpha': infilled_mpi_alpha,
        }
        return output_batch
