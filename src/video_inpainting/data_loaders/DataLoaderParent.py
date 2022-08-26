# Shree KRISHNAya Namaha
# A parent class for all data-loaders
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import abc
from pathlib import Path

import torch.utils.data

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class DataLoaderParent(torch.utils.data.Dataset):

    def __init__(self):
        super(DataLoaderParent, self).__init__()
        pass

    @abc.abstractmethod
    def load_test_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        pass

    @staticmethod
    @abc.abstractmethod
    def post_process_tensor(tensor: torch.Tensor):
        pass
