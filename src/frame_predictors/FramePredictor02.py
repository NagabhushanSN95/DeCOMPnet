# Shree KRISHNAya Namaha
# Extended from FramePredictor01.py. True flow12 is used instead of estimating and predicting
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import time
import datetime
import traceback
from collections import OrderedDict

import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot
import torch
import torch.nn.functional as F

from utils import MpiUtils, CommonUtils
from utils.WarperPytorch import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class FramePredictor(torch.nn.Module):
    def __init__(self, configs: dict, feature_extractor, local_flow_estimator, local_flow_predictor, 
                 infilling_flow_predictor):
        super().__init__()
        self.configs = configs
        self.feature_extractor = feature_extractor
        self.local_flow_estimator = local_flow_estimator
        self.local_flow_predictor = local_flow_predictor
        self.infilling_flow_predictor = infilling_flow_predictor
        self.warper = Warper(configs['device'])
        self.num_mpi_planes = self.configs['frame_predictor']['num_mpi_planes']
        self.num_pred_frames = self.configs['frame_predictor']['num_pred_frames']
        self.num_infilling_iterations = self.configs['frame_predictor']['num_infilling_iterations']
        self.device = CommonUtils.get_device(configs['device'])

        assert self.num_pred_frames == 1
        return

    def forward(self, input_batch: dict):
        b, c, h, w = input_batch['frame1'].shape

        # Warp frame1 to frame2 using flow12
        warped_frame2, warped_mask2 = self.warper.bilinear_splatting(input_batch['frame1'], None, input_batch['depth1'],
                                                                     input_batch['flow12'], None, is_image=True)
        warped_depth2, _ = self.warper.bilinear_splatting(input_batch['depth1'], None, input_batch['depth1'],
                                                          input_batch['flow12'], None, is_image=False)

        depth2a = warped_depth2 + 1000 * (1 - warped_mask2)
        min_depth = depth2a.reshape((b, h*w)).min(dim=1)[0]
        max_depth = torch.clip(warped_depth2[:, :, 6:-6].reshape((b, (h - 12)*w)).max(dim=1)[0], min=0, max=1000)
        depth2_planes = MpiUtils.get_depth_planes_tr(self.num_mpi_planes, min_depth, max_depth)
        warped_mpi2_alpha, warped_mpi2_depth_planes, warped_mpi2_rgb = MpiUtils.create_mpi_tr(
            warped_depth2, depth2_planes, [warped_frame2])
        warped_mpi2_alpha = warped_mpi2_alpha * warped_mask2[:, :, :, :, None]

        infilling_inputs = {
            'mpi_rgb': warped_mpi2_rgb,
            'mpi_alpha': warped_mpi2_alpha,
        }
        with torch.cuda.amp.autocast():
            infilling_outputs = self.infill_mpi(infilling_inputs)
        infilled_mpi2_rgb = infilling_outputs['infilled_mpi_rgb']
        infilled_mpi2_alpha = infilling_outputs['infilled_mpi_alpha']

        predicted_frame2, mask2 = MpiUtils.alpha_compositing(infilled_mpi2_rgb, infilled_mpi2_alpha)
        output_batch = {
            'predicted_frames': [predicted_frame2],
            'predicted_frames_mask': [mask2],
        }
        return output_batch

    def infill_mpi(self, input_batch):
        mpi_rgb = input_batch['mpi_rgb']
        mpi_alpha = input_batch['mpi_alpha']

        for i in range(self.num_infilling_iterations):
            mask = MpiUtils.alpha2mask(mpi_alpha)
            infill_input_batch = {
                'mpi_rgb': mpi_rgb,
                'mpi_alpha': mpi_alpha,
                'mask': mask,
            }
            infill_output_batch = self.infill_one_iter(infill_input_batch)
            mpi_rgb = infill_output_batch['infilled_mpi_rgb']
            mpi_alpha = infill_output_batch['infilled_mpi_alpha']

        pred_frame, pred_mask = MpiUtils.alpha_compositing(mpi_rgb, mpi_alpha)
        output_batch = {
            'infilled_mpi_rgb': mpi_rgb,
            'infilled_mpi_alpha': mpi_alpha,
            'pred_frame': pred_frame,
            'pred_mask': pred_mask,
        }
        return output_batch

    def infill_one_iter(self, input_batch: dict):
        flow_predictor_output = self.infilling_flow_predictor(input_batch)

        warper_inputs = {
            'mpi_rgb': input_batch['mpi_rgb'],
            'mpi_alpha': input_batch['mpi_alpha'],
            'disoccluded_flow': flow_predictor_output['disoccluded_flow'],
        }
        warper_outputs = self.warper.bilinear_interpolation_mpi_flow2d(warper_inputs)
        mask = input_batch['mask'][:, :, :, :, None]
        infilled_mpi_rgb = mask * input_batch['mpi_rgb'] + (1 - mask) * warper_outputs['warped_rgb']
        infilled_mpi_alpha = mask * input_batch['mpi_alpha'] + (1 - mask) * warper_outputs['warped_alpha']

        del flow_predictor_output['disoccluded_flow'], warper_outputs['warped_rgb'], warper_outputs['warped_alpha']

        output_batch = {
            'infilled_mpi_rgb': infilled_mpi_rgb,
            'infilled_mpi_alpha': infilled_mpi_alpha,
        }
        return output_batch

    def load_weights(self, flow_weights_path: Path, inpainting_weights_path: Path):
        checkpoint_state = torch.load(flow_weights_path, map_location=self.device)
        iter_num = checkpoint_state['iteration_num']
        self.load_sub_network_weights(self.feature_extractor, 'feature_extractor', checkpoint_state['model_state_dict'])
        self.load_sub_network_weights(self.local_flow_estimator, 'flow_estimator', checkpoint_state['model_state_dict'])
        print(f'Loaded Model {flow_weights_path} trained for {iter_num} iterations')

        checkpoint_state = torch.load(inpainting_weights_path, map_location=self.device)
        iter_num = checkpoint_state['iteration_num']
        self.load_sub_network_weights(self.infilling_flow_predictor, 'flow_predictor', checkpoint_state['model_state_dict'])
        print(f'Loaded Model {inpainting_weights_path} trained for {iter_num} iterations')
        return

    @staticmethod
    def load_sub_network_weights(network, network_name, weights_dict: OrderedDict):
        selected_weights = OrderedDict()
        for weight_name in weights_dict.keys():
            if weight_name.startswith(network_name):
                new_weight_name = weight_name[len(network_name)+1:]
                selected_weights[new_weight_name] = weights_dict[weight_name]
        network.load_state_dict(selected_weights)
        return
