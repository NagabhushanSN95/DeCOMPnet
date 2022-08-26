# Shree KRISHNAya Namaha
# A parent class for all data-loaders
# Author: Nagabhushan S N
# Last Modified: 26/08/2022
import abc
import time
import datetime
import traceback
from typing import Optional

import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path

import torch.utils.data
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DataLoaderParent(torch.utils.data.Dataset):

    def __init__(self):
        super(DataLoaderParent, self).__init__()
        pass

    @abc.abstractmethod
    def load_test_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        pass

    @abc.abstractmethod
    def load_prediction_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        pass

    @abc.abstractmethod
    def load_generation_data(self, video_name: str, seq_num: int, frame1_num: int, pw_dirname):
        pass

    @staticmethod
    @abc.abstractmethod
    def post_process_tensor(tensor: torch.Tensor):
        pass
