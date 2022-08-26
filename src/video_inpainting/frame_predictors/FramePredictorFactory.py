# Shree KRISHNAya Namaha
# A Factory method that returns a frame predictor
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import importlib.util
import inspect


def get_frame_predictor(configs: dict, flow_predictor):
    filename = configs['video_inpainting']['frame_predictor']['name']
    classname = f'FramePredictor'
    frame_predictor = None
    module = importlib.import_module(f'video_inpainting.frame_predictors.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            frame_predictor = candidate_class[1](configs, flow_predictor)
            break
    if frame_predictor is None:
        raise RuntimeError(f'Unknown frame_predictor: {filename}')
    return frame_predictor
