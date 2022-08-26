# Shree KRISHNAya Namaha
# Common Utility Functions
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_device(device: str):
    """
    Returns torch device object
    :param device: cpu/gpu0/gpu1
    :return:
    """
    if device == 'cpu':
        device = torch.device('cpu')
    elif device.startswith('gpu') and torch.cuda.is_available():
        gpu_num = int(device[3:])
        device = torch.device(f'cuda:{gpu_num}')
    else:
        device = torch.device('cpu')
    return device


def move_to_device(data: Union[dict, list, torch.Tensor], device):
    moved_data = None
    if isinstance(data, torch.Tensor):
        if device.type == 'cpu':
            moved_data = data.to(device)
        else:
            moved_data = data.cuda(device, non_blocking=True)
    elif isinstance(data, list):
        moved_data = []
        for data_element in data:
            moved_data.append(move_to_device(data_element, device))
    elif isinstance(data, dict):
        moved_data = {}
        for k in data:
            moved_data[k] = move_to_device(data[k], device)
    return moved_data
