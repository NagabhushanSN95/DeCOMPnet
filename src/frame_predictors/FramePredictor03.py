# Shree KRISHNAya Namaha
# Extended from FramePredictor01.py. True frame2 is used to infill disocclusions
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
        min_depth = input_batch['depth1'].reshape((b, h*w)).min(dim=1)[0]
        max_depth = torch.clip(input_batch['depth1'].reshape((b, h*w)).max(dim=1)[0], min=0, max=1000)
        depth1_planes = MpiUtils.get_depth_planes_tr(self.num_mpi_planes, min_depth, max_depth)
        warper_inputs = {
            'frame1': input_batch['frame0'],
            'depth1': input_batch['depth0'],
            'transformation1': input_batch['transformation0'],
            'transformation2': input_batch['transformation1'],
            'intrinsic1': input_batch['intrinsic0'],
            'intrinsic2': input_batch['intrinsic1'],
            'depth2_planes': depth1_planes,
        }
        warper_outputs = self.warper.forward_warp_frame2mpi(warper_inputs)
        warped_mpi0_rgb = warper_outputs['warped_mpi2_rgb']
        warped_mpi0_alpha = warper_outputs['warped_mpi2_alpha']

        mpi1_alpha, mpi1_depth_planes, mpi1_rgb = MpiUtils.create_mpi_tr(input_batch['depth1'], depth1_planes,
                                                                         [input_batch['frame1']])

        # Estimate past local flow
        flow_inputs = {
            'mpi0_rgb': warped_mpi0_rgb,
            'mpi0_alpha': warped_mpi0_alpha,
            'mpi1_rgb': mpi1_rgb,
            'mpi1_alpha': mpi1_alpha,
            'depth1_planes': mpi1_depth_planes,
        }
        with torch.cuda.amp.autocast():
            flow_outputs = self.estimate_past_local_flow(flow_inputs)
        local_mpi_flow10 = flow_outputs['local_mpi_flow10']
        # local_mpi_flow10_mask = flow_outputs['local_mpi_flow10_mask']
        local_flow10, local_flow10_mask = MpiUtils.alpha_compositing(local_mpi_flow10, mpi1_alpha)

        output_batch = {
            'predicted_frames': [],
            'predicted_frames_mask': [],
        }

        # Delete redundant tensors to free memory
        del warped_mpi0_rgb, warped_mpi0_alpha, mpi1_depth_planes, local_flow10_mask

        for i in range(self.num_pred_frames):
            # Predict future flow
            flow_predictor_input = {
                'local_flow10': local_flow10,
                # 'local_mpi_flow10_mask': local_mpi_flow10_mask,
                'num_past_steps': self.num_pred_frames + 1,
                'num_future_steps': i + 1,
            }
            flow_predictor_output = self.local_flow_predictor(flow_predictor_input)
            local_flow12 = flow_predictor_output['local_flow12_predicted']
            # local_flow12_mask = flow_predictor_output['local_flow12_predicted_mask']

            # Warp mpi1 to mpi2 using local_mpi_flow12
            warper_inputs = {
                'frame1': input_batch['frame1'],
                'depth1': input_batch['depth1'],
                'local_flow12': local_flow12,
                # 'local_flow12_mask': local_flow12_mask,
                'transformation1': input_batch['transformation1'],
                'transformation2': input_batch['transformation2'][i],
                'intrinsic1': input_batch['intrinsic1'],
                'intrinsic2': input_batch['intrinsic2'][i],
                'num_mpi_planes': self.num_mpi_planes,
            }
            warper_outputs = self.warper.forward_warp_frame2mpi(warper_inputs)
            warped_mpi2_rgb, warped_mpi2_alpha = warper_outputs['warped_mpi2_rgb'], warper_outputs['warped_mpi2_alpha']

            # Ground truth infilling
            warped_frame2, warped_frame2_mask = MpiUtils.alpha_compositing(warped_mpi2_rgb, warped_mpi2_alpha)
            true_frame2 = input_batch['frame2'][i]
            infilled_frame2 = warped_frame2_mask * warped_frame2 + (1 - warped_frame2_mask) * true_frame2

            output_batch['predicted_frames'].append(infilled_frame2)
            output_batch['predicted_frames_mask'].append(warped_frame2_mask)
        return output_batch

    def estimate_past_local_flow(self, input_batch):
        # Extract features
        feature_extractor_input = {
            'mpi_rgb': input_batch['mpi1_rgb'],
            'mpi_alpha': input_batch['mpi1_alpha'],
        }
        feature_extractor_output = self.feature_extractor(feature_extractor_input)
        mpi1_features = feature_extractor_output['mpi_features']
        feature_extractor_input = {
            'mpi_rgb': input_batch['mpi0_rgb'],
            'mpi_alpha': input_batch['mpi0_alpha'],
        }
        feature_extractor_output = self.feature_extractor(feature_extractor_input)
        mpi0_features = feature_extractor_output['mpi_features']

        # Estimate flow
        flow_estimator_input = {
            'mpi1_features': mpi1_features,
            'mpi2_features': mpi0_features,
            'mpi1_alpha': input_batch['mpi1_alpha'],
            'mpi2_alpha': input_batch['mpi0_alpha'],
        }
        flow_estimator_output = self.local_flow_estimator(flow_estimator_input)
        local_mpi_flow10 = flow_estimator_output['estimated_mpi_flows12'][0][0]
        # local_mpi_flow10_mask = flow_estimator_output['estimated_mpi_flows12'][0][1]

        # Take expectation of z-flow distribution
        local_mpi_flow10xy = local_mpi_flow10[:, :2]
        local_mpi_flow10z = torch.sum(local_mpi_flow10[:, 2:6] * input_batch['depth1_planes'], dim=1, keepdim=True) - input_batch['depth1_planes']
        local_mpi_flow10 = torch.cat([local_mpi_flow10xy, local_mpi_flow10z], dim=1) * input_batch['mpi1_alpha']

        output_batch = {
            'local_mpi_flow10': local_mpi_flow10,
            # 'local_mpi_flow10_mask': local_mpi_flow10_mask,
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
