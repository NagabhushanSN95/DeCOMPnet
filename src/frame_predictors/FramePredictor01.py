# Shree KRISHNAya Namaha
#
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
        return

    def forward(self, input_batch: dict, return_intermediate_results: bool = False):
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

            infilling_inputs = {
                'mpi_rgb': warped_mpi2_rgb,
                'mpi_alpha': warped_mpi2_alpha,
            }
            with torch.cuda.amp.autocast():
                infilling_outputs = self.infill_mpi(infilling_inputs)
            infilled_mpi2_rgb = infilling_outputs['infilled_mpi_rgb']
            infilled_mpi2_alpha = infilling_outputs['infilled_mpi_alpha']

            predicted_frame2, mask2 = MpiUtils.alpha_compositing(infilled_mpi2_rgb, infilled_mpi2_alpha)
            output_batch['predicted_frames'].append(predicted_frame2)
            output_batch['predicted_frames_mask'].append(mask2)

        if return_intermediate_results:
            assert self.num_pred_frames == 1
            intermediate_inputs = {
                # 'mpi_rgb': mpi1_rgb,
                # 'mpi1_alpha': mpi1_alpha,
                'frame1': input_batch['frame1'],
                'depth1': input_batch['depth1'],
                'local_flow12': local_flow12,
                'total_flow_warped_mpi2_rgb': warped_mpi2_rgb,
                'total_flow_warped_mpi2_alpha': warped_mpi2_alpha,
            }
            intermediate_outputs = self.compute_intermediate_results(intermediate_inputs)
            output_batch.update(intermediate_outputs)
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
            # disoccluded_flow = infill_output_batch['disoccluded_flow']

        pred_frame, pred_mask = MpiUtils.alpha_compositing(mpi_rgb, mpi_alpha)
        output_batch = {
            # 'disoccluded_flow': disoccluded_flow,
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
            # 'disoccluded_flow': flow_predictor_output['disoccluded_flow'],
            'infilled_mpi_rgb': infilled_mpi_rgb,
            'infilled_mpi_alpha': infilled_mpi_alpha,
        }
        return output_batch
    
    def compute_intermediate_results(self, input_batch):
        output_batch = {
            'local_flow12': input_batch['local_flow12'],
            'total_flow_warped_mpi2_rgb': input_batch['total_flow_warped_mpi2_rgb'],
            'total_flow_warped_mpi2_alpha': input_batch['total_flow_warped_mpi2_alpha'],
        }
        local_flow_warped_frame2 = self.warper.bilinear_splatting(input_batch['frame1'], None, input_batch['depth1'], 
                                                         input_batch['local_flow12'][:, :2], None)[0]
        total_flow_warped_frame2 = MpiUtils.alpha_compositing(input_batch['total_flow_warped_mpi2_rgb'], 
                                                              input_batch['total_flow_warped_mpi2_alpha'])[0]
        
        output_batch['local_flow_warped_frame2'] = local_flow_warped_frame2
        output_batch['total_flow_warped_frame2'] = total_flow_warped_frame2
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
