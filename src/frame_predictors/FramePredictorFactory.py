# Shree KRISHNAya Namaha
# A Factory method that returns a Model
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


def get_frame_predictor(configs: dict, feature_extractor, local_flow_estimator, local_flow_predictor, infilling_flow_estimator):
    filename = configs['frame_predictor']['name']
    classname = f'FramePredictor'
    model = None
    module = importlib.import_module(f'frame_predictors.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            model = candidate_class[1](configs, feature_extractor, local_flow_estimator, local_flow_predictor, infilling_flow_estimator)
            break
    if model is None:
        raise RuntimeError(f'Unknown model: {filename}')
    return model
