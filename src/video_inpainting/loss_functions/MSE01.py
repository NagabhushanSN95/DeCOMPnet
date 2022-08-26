# Shree KRISHNAya Namaha
# MSE between predicted frames and ground truth frames
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import torch

from video_inpainting.loss_functions.LossFunctionParent01 import LossFunctionParent


class MSE(LossFunctionParent):
    def __init__(self, configs: dict, loss_configs: dict):
        self.configs = configs
        self.loss_configs = loss_configs
        return

    def compute_loss(self, input_dict: dict, output_dict: dict):
        true_frame = input_dict['target_frame']
        pred_frame = output_dict['pred_frame']
        mse = self.compute_mse(true_frame, pred_frame)
        return mse

    @staticmethod
    def compute_mse(true_frame, pred_frame):
        mse = torch.mean(torch.square(true_frame - pred_frame))
        return mse
