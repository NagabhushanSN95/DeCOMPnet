# Shree KRISHNAya Namaha
# Utility functions related to MPI
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from pathlib import Path

import numpy
import torch

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def create_mpi_tr(depth, depth_planes, data2d_list):
    nearest_plane_index = (depth_planes[:, None, None, None, :] - depth[:, :, :, :, None]).abs().argmin(dim=4)
    b, _, h, w = depth.shape
    d = depth_planes.shape[1]
    mpi_alpha = torch.zeros(size=(b, 1, h, w, d)).to(depth)
    mpi_alpha[torch.arange(b)[:, None, None], :, numpy.arange(h)[None, :, None], numpy.arange(w)[None, None, :], nearest_plane_index] = 1
    depth_mpi_planes = torch.broadcast_to(depth_planes[:, None, None, None], mpi_alpha.shape)
    return_list = [mpi_alpha, depth_mpi_planes]

    for data2d in data2d_list:
        data3d = data2d[:, :, :, :, None] * mpi_alpha
        return_list.append(data3d)
    return return_list


def alpha2visibility(mpi_alpha):
    first_plane_vis = torch.ones((*mpi_alpha.shape[:-1], 1)).to(mpi_alpha)
    visibility = torch.cumprod(torch.cat([first_plane_vis, 1. - mpi_alpha + 1e-10], -1), -1)[..., :-1]
    return visibility


def alpha2mask(mpi_alpha):
    mask = (torch.sum(mpi_alpha, dim=4) >= 1).float()
    return mask


def alpha_compositing(mpi_rgb, mpi_alpha):
    visibility = alpha2visibility(mpi_alpha)
    weights = mpi_alpha * visibility
    frame = torch.sum(weights * mpi_rgb, dim=4)
    mask = torch.sum(weights * mpi_alpha, dim=4)
    return frame, mask


def get_depth_planes(num_mpi_planes, min_depth, max_depth):
    min_disp = 1 / (min_depth + 1e-3)
    max_disp = 1 / (max_depth + 1e-3)
    disp_planes = numpy.linspace(min_disp, max_disp, num_mpi_planes)
    depth_planes = 1 / disp_planes
    return depth_planes


def get_depth_planes_tr(num_mpi_planes: int, min_depth: torch.Tensor, max_depth: torch.Tensor):
    min_disp = 1 / (min_depth + 1e-3)
    max_disp = 1 / (max_depth + 1e-3)
    disp_planes = torch.arange(num_mpi_planes)[None].to(min_depth) * ((max_disp - min_disp) / (num_mpi_planes - 1)) + min_disp
    depth_planes = 1 / disp_planes
    return depth_planes
