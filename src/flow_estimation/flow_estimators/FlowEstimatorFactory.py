# Shree KRISHNAya Namaha
# A Factory method that returns a flow estimator
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import importlib.util
import inspect


def get_flow_estimator(configs: dict):
    filename = configs['flow_estimation']['flow_estimator']['name']
    classname = f'FlowEstimator'
    model = None
    module = importlib.import_module(f'flow_estimation.flow_estimators.{filename}')
    candidate_classes = inspect.getmembers(module, inspect.isclass)
    for candidate_class in candidate_classes:
        if candidate_class[0] == classname:
            model = candidate_class[1](configs)
            break
    if model is None:
        raise RuntimeError(f'Unknown model: {filename}')
    return model
