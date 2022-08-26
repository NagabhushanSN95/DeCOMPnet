# Shree KRISHNAya Namaha
# A Factory method that returns a Local Flow Predictor
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import time
import datetime
import importlib.util
import inspect
import time
import traceback
from typing import Optional

import numpy
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot


def get_local_flow_predictor(configs: dict):
    filename = configs['local_flow_predictor']['name']
    classname = f'LocalFlowPredictor'
    model = None
    module = importlib.import_module(f'local_flow_predictors.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            model = candidate_class[1](configs)
            break
    if model is None:
        raise RuntimeError(f'Unknown model: {filename}')
    return model
