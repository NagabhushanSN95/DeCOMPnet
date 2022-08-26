# Shree KRISHNAya Namaha
# A Factory method that returns a feature extractor
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import importlib.util
import inspect


def get_feature_extractor(configs: dict):
    filename = configs['flow_estimation']['feature_extractor']['name']
    classname = f'FeatureExtractor'
    model = None
    module = importlib.import_module(f'flow_estimation.feature_extractors.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            model = candidate_class[1](configs)
            break
    if model is None:
        raise RuntimeError(f'Unknown model: {filename}')
    return model
