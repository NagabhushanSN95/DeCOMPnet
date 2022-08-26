# Shree KRISHNAya Namaha
# A Factory method that returns a flow_predictor
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import importlib.util
import inspect


def get_flow_predictor(configs: dict):
    filename = configs['video_inpainting']['flow_predictor']['name']
    classname = f'FlowPredictor'
    flow_predictor = None
    module = importlib.import_module(f'video_inpainting.flow_predictors.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            flow_predictor = candidate_class[1](configs)
            break
    if flow_predictor is None:
        raise RuntimeError(f'Unknown flow_predictor: {filename}')
    return flow_predictor
