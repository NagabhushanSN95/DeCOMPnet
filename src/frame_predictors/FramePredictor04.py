# Shree KRISHNAya Namaha
# Mix of FramePredictor02.py and FramePredictor03.py. Ground truth flow and infilling.
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
        # Warp frame1 to frame2 using flow12
        warped_frame2, warped_mask2 = self.warper.bilinear_splatting(input_batch['frame1'], None, input_batch['depth1'],
                                                                     input_batch['flow12'], None, is_image=True)

        # Ground truth infilling
        true_frame2 = input_batch['frame2'][0]
        infilled_frame2 = warped_mask2 * warped_frame2 + (1 - warped_mask2) * true_frame2

        output_batch = {
            'predicted_frames': [infilled_frame2],
            'predicted_frames_mask': [warped_mask2],
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
