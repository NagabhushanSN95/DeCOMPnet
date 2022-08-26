# Shree KRISHNAya Namaha
# Abstract parent class
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import abc


class LossFunctionParent:
    @abc.abstractmethod
    def __init__(self, configs: dict, loss_configs: dict):
        pass

    @abc.abstractmethod
    def compute_loss(self, input_dict: dict, output_dict: dict):
        pass
