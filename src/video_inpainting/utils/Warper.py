# Shree KRISHNAya Namaha
# Differentiable warper implemented in PyTorch. Warping is done on batches.
# Tested on PyTorch 1.8.1
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional

import numpy
import skimage.io
import torch
import torch.nn.functional as F

from utils import CommonUtils
from utils.MpiUtils import get_depth_planes, get_depth_planes_tr


class Warper:
    def __init__(self, device: str = 'gpu0'):
        self.device = CommonUtils.get_device(device)
        return

    def bilinear_interpolation_mpi_flow2d(self, input_batch: dict) -> dict:
        mpi_rgb = input_batch['mpi_rgb']
        mpi_alpha = input_batch['mpi_alpha']
        # mpi_depth = input_batch['mpi_depth']
        disoccluded_flow = input_batch['disoccluded_flow']

        b, c, h, w, d = mpi_rgb.shape
        grid = self.create_grid_mpi(b, h, w, d).to(mpi_rgb)

        mpi_rgb_offset = F.pad(mpi_rgb, [0, 0, 1, 1, 1, 1])
        mpi_alpha_offset = F.pad(mpi_alpha, [0, 0, 1, 1, 1, 1])
        # mpi_depth_offset = F.pad(mpi_depth, [0, 0, 1, 1, 1, 1])
        bi = torch.arange(b)[:, None, None, None]
        di = torch.arange(d)[None, None, None, :]

        trans_pos = disoccluded_flow + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        prox_weight_sum = prox_weight_nw + prox_weight_sw + prox_weight_ne + prox_weight_se
        prox_weight_nw = prox_weight_nw / prox_weight_sum
        prox_weight_sw = prox_weight_sw / prox_weight_sum
        prox_weight_ne = prox_weight_ne / prox_weight_sum
        prox_weight_se = prox_weight_se / prox_weight_sum

        weight_nw = torch.moveaxis(prox_weight_nw, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_sw = torch.moveaxis(prox_weight_sw, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_ne = torch.moveaxis(prox_weight_ne, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])
        weight_se = torch.moveaxis(prox_weight_se, [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])

        f2_nw = mpi_rgb_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0], di]
        f2_sw = mpi_rgb_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], di]
        f2_ne = mpi_rgb_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], di]
        f2_se = mpi_rgb_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], di]
        warped_rgb = weight_nw * f2_nw + weight_sw * f2_sw + weight_ne * f2_ne + weight_se * f2_se
        warped_rgb = torch.moveaxis(warped_rgb, [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])

        a2_nw = mpi_alpha_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0], di]
        a2_sw = mpi_alpha_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], di]
        a2_ne = mpi_alpha_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], di]
        a2_se = mpi_alpha_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], di]
        warped_alpha = weight_nw * a2_nw + weight_sw * a2_sw + weight_ne * a2_ne + weight_se * a2_se
        warped_alpha = torch.moveaxis(warped_alpha, [0, 1, 2, 3, 4], [0, 2, 3, 4, 1])

        # d2_nw = mpi_depth_offset[bi, :, trans_pos_floor[:, 1], trans_pos_floor[:, 0], di]
        # d2_sw = mpi_depth_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_floor[:, 0], di]
        # d2_ne = mpi_depth_offset[bi, :, trans_pos_floor[:, 1], trans_pos_ceil[:, 0], di]
        # d2_se = mpi_depth_offset[bi, :, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0], di]
        # warped_depth = weight_nw * d2_nw + weight_sw * d2_sw + weight_ne * d2_ne + weight_se * d2_se
        # warped_depth = torch.moveaxis(warped_depth, [0, 1, 2, 3], [0, 2, 3, 1])

        output_batch = {
            'warped_rgb': warped_rgb,
            'warped_alpha': warped_alpha,
            # 'warped_depth': warped_depth,
        }
        return output_batch

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def create_grid_mpi(b, h, w, d):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None, :, :, :, None].repeat([b, 1, 1, 1, d])
        return batch_grid
