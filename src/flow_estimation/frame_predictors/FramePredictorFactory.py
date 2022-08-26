# Shree KRISHNAya Namaha
# A Factory method that returns a frame predictor
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import importlib.util
import inspect


def get_frame_predictor(configs: dict, feature_extractor, flow_estimator):
    filename = configs['flow_estimation']['frame_predictor']['name']
    classname = f'FramePredictor'
    model = None
    module = importlib.import_module(f'flow_estimation.frame_predictors.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            model = candidate_class[1](configs, feature_extractor, flow_estimator)
            break
    if model is None:
        raise RuntimeError(f'Unknown model: {filename}')
    return model
