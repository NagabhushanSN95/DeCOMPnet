# Shree KRISHNAya Namaha
# A Factory method that returns a Data Loader
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import importlib.util
import inspect
from pathlib import Path
from typing import Optional


def get_data_loader(configs: dict, data_dirpath: Path, frames_datapath: Optional[str]):
    filename = configs['video_inpainting']['data_loader']['name']
    classname = 'DataLoader'
    data_loader = None
    module = importlib.import_module(f'video_inpainting.data_loaders.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            data_loader = candidate_class[1](configs, data_dirpath, frames_datapath)
            break
    if data_loader is None:
        raise RuntimeError(f'Unknown data loader: {filename}')
    return data_loader
