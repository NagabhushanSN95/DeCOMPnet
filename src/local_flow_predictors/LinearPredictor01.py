# Shree KRISHNAya Namaha
# w_{n, n+1} = (-1/k) * w_{n, n-k}
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class LocalFlowPredictor:
    def __init__(self, configs: dict):
        self.configs = configs
        return

    def __call__(self, input_dict: dict):
        return self.predict_future_flow(input_dict)

    @staticmethod
    def predict_future_flow(input_dict: dict):
        flow10 = input_dict['local_flow10']
        # flow10_mask = input_dict['local_flow10_mask']
        scaling_factor = - input_dict['num_future_steps'] / input_dict['num_past_steps']
        flow12 = scaling_factor * flow10
        result_dict = {
            'local_flow12_predicted': flow12,
            # 'local_flow12_predicted_mask': flow10_mask,
        }
        return result_dict
