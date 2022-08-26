# Shree KRISHNAYa Namaha
# Modified from OursBlender01.py.
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import random
from pathlib import Path
from typing import Optional

import numpy
import pandas
import skimage.io
import torch
from tqdm import tqdm
from matplotlib import pyplot

from data_loaders.DataLoaderParent import DataLoaderParent
from utils import MpiUtils


class DataLoader(DataLoaderParent):

    def __init__(self, configs: dict, data_dirpath: Path, frames_datapath: Optional[Path]):
        super(DataLoader, self).__init__()
        self.configs = configs
        self.dataroot = data_dirpath
        if frames_datapath is not None:
            video_data = pandas.read_csv(frames_datapath)
            self.video_names = video_data.to_numpy().tolist()
        else:
            self.video_names = None
        # self.size = configs['data_loader']['patch_size']
        self.num_mpi_planes = configs['frame_predictor']['num_mpi_planes']
        self.num_pred_frames = configs['frame_predictor']['num_pred_frames']
        self.num_steps = self.num_pred_frames + 1
        if 'resolution_suffix' in self.configs['data_loader']:
            self.resolution_suffix = self.configs['data_loader']['resolution_suffix']
        else:
            self.resolution_suffix = ''
        self.frame_resolution = self.get_frame_resolution(self.resolution_suffix)
        return

    @staticmethod
    def get_frame_resolution(resolution_suffix):
        h, w = 436, 1024
        if resolution_suffix == '':
            frame_resolution = h, w
        elif '_down' in resolution_suffix:
            scale = int(resolution_suffix[5:])
            frame_resolution = h // scale, w // scale
        elif '_up' in resolution_suffix:
            scale = int(resolution_suffix[3:])
            frame_resolution = h * scale, w * scale
        else:
            raise RuntimeError(f'Unknown resolution_suffix: {resolution_suffix}')
        return frame_resolution

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        data_dict = self.load_training_data(index)
        return data_dict

    def load_testing_data(self, video_name: str, seq_name: int, pred_frame_num: int, padding: tuple = None):
        psp = None
        flip = False

        frame1_num = pred_frame_num - 1
        frame0_num = frame1_num - self.num_steps
        frame0_path = self.dataroot / f'all/RenderedData/{video_name}/rgb{self.resolution_suffix}/{seq_name}/{frame0_num:04}.npy'
        frame1_path = self.dataroot / f'all/RenderedData/{video_name}/rgb{self.resolution_suffix}/{seq_name}/{frame1_num:04}.npy'
        depth0_path = self.dataroot / f'all/RenderedData/{video_name}/depth{self.resolution_suffix}/{frame0_num:04}.npy'
        depth1_path = self.dataroot / f'all/RenderedData/{video_name}/depth{self.resolution_suffix}/{frame1_num:04}.npy'
        transformation_path = self.dataroot / f'all/RenderedData/{video_name}/TransformationMatrices.csv'
        intrinsic_path = self.dataroot / f'all/RenderedData/{video_name}/CameraIntrinsics.csv'

        frame0 = self.read_npy_file(frame0_path)
        frame1 = self.read_npy_file(frame1_path)
        depth0 = self.read_npy_file(depth0_path)
        depth1 = self.read_npy_file(depth1_path)
        transformation_matrices = numpy.genfromtxt(transformation_path, delimiter=',', dtype=numpy.float32).reshape((-1, 4, 4))
        transformation0 = transformation_matrices[frame0_num]
        transformation1 = transformation_matrices[frame1_num]
        transformation2 = [transformation_matrices[pred_frame_num + i] for i in range(self.num_pred_frames)]
        intrinsic_matrices = numpy.genfromtxt(intrinsic_path, delimiter=',', dtype=numpy.float32).reshape((-1, 3, 3))
        intrinsic0 = intrinsic_matrices[frame0_num]
        intrinsic1 = intrinsic_matrices[frame1_num]
        intrinsic2 = [intrinsic_matrices[pred_frame_num + i] for i in range(self.num_pred_frames)]

        frame1 = self.preprocess_frame(frame1, psp, padding, flip)
        frame0 = self.preprocess_frame(frame0, psp, padding, flip)
        depth1 = self.preprocess_depth(depth1, psp, padding, flip)
        depth0 = self.preprocess_depth(depth0, psp, padding, flip)

        data_dict = {
            'frame0': torch.from_numpy(frame0),
            'frame1': torch.from_numpy(frame1),
            'depth0': torch.from_numpy(depth0),
            'depth1': torch.from_numpy(depth1),
            'transformation0': torch.from_numpy(transformation0),
            'transformation1': torch.from_numpy(transformation1),
            'transformation2': [torch.from_numpy(transformation) for transformation in transformation2],
            'intrinsic0': torch.from_numpy(intrinsic0),
            'intrinsic1': torch.from_numpy(intrinsic1),
            'intrinsic2': [torch.from_numpy(intrinsic) for intrinsic in intrinsic2],
        }

        if self.configs['data_loader']['load_true_flow']:
            flow_path = self.dataroot / f'all/RenderedData/{video_name}/flow_data/flow{self.resolution_suffix}/{frame1_num:04}.flo'
            flow12 = self.read_flow(flow_path)
            flow12 = self.preprocess_flow(flow12, psp, padding, flip)
            data_dict['flow12'] = torch.from_numpy(flow12)

        if self.configs['data_loader']['load_target_frames']:
            target_frame_paths = [self.dataroot / f'all/RenderedData/{video_name}/rgb{self.resolution_suffix}/{seq_name}/{pred_frame_num + i:04}.npy' for i in range(self.num_pred_frames)]
            target_frames = [self.read_npy_file(path) for path in target_frame_paths]
            target_frames = [self.preprocess_frame(frame, psp, padding, flip) for frame in target_frames]
            data_dict['frame2'] = [torch.from_numpy(frame) for frame in target_frames]
        return data_dict

    @staticmethod
    def read_npy_file(path: Path, mmap_mode: str = None):
        if path.suffix == '.npy':
            try:
                array = numpy.load(path.as_posix(), mmap_mode=mmap_mode)
            except Exception as e:
                print(f'Error in reading file: {path.as_posix()}')
                raise e
        else:
            raise RuntimeError(f'Unknown array format: {path.as_posix()}')
        return array

    @staticmethod
    def read_flow(path: Path):
        with open(path.as_posix(), 'rb') as f:
            magic = numpy.fromfile(f, numpy.float32, count=1)[0]
            forward_flow = None

            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
            else:
                w = numpy.fromfile(f, numpy.int32, count=1)[0]
                h = numpy.fromfile(f, numpy.int32, count=1)[0]
                forward_flow = numpy.fromfile(f, numpy.float32, count=2 * w * h)
                # reshape data into 3D array (columns, rows, channels)
                forward_flow = numpy.resize(forward_flow, (h, w, 2))
        return forward_flow

    def preprocess_frame(self, frame: numpy.ndarray, psp: Optional[list], padding: Optional[tuple], flip: bool):
        if psp is not None:
            y1, x1 = psp
            y2, x2 = y1 + self.size[0], x1 + self.size[1]
            frame = frame[y1:y2, x1:x2]
        if padding is not None:
            py, px = padding
            frame = numpy.pad(frame, pad_width=((py, py), (px, px), (0, 0)), mode='constant', constant_values=0)
        if flip:
            frame = numpy.flip(frame, axis=1)
        norm_frame = frame.astype('float32') / 255
        cf_frame = numpy.moveaxis(norm_frame, [0, 1, 2], [1, 2, 0])
        return cf_frame

    def preprocess_mask(self, mask: numpy.ndarray, psp: Optional[list], padding: Optional[tuple], flip: bool):
        if psp is not None:
            y1, x1 = psp
            y2, x2 = y1 + self.size[0], x1 + self.size[1]
            mask = mask[y1:y2, x1:x2]
        if padding is not None:
            py, px = padding
            mask = numpy.pad(mask, pad_width=((py, py), (px, px)), mode='constant', constant_values=0)
        if flip:
            mask = numpy.flip(mask, axis=1)
        cf_mask = mask[None].astype('float32')
        return cf_mask

    def preprocess_depth(self, depth: numpy.ndarray, psp: Optional[list], padding: Optional[tuple], flip: bool):
        if psp is not None:
            y1, x1 = psp
            y2, x2 = y1 + self.size[0], x1 + self.size[1]
            depth = depth[y1:y2, x1:x2]
        if padding is not None:
            py, px = padding
            depth = numpy.pad(depth, pad_width=((py, py), (px, px)), mode='constant', constant_values=1000)
        if flip:
            depth = numpy.flip(depth, axis=1)
        cf_depth = depth[None].astype('float32')
        return cf_depth

    def preprocess_flow(self, flow: numpy.ndarray, psp: Optional[list], padding: Optional[tuple], flip: bool):
        if psp is not None:
            y1, x1 = psp
            y2, x2 = y1 + self.size[0], x1 + self.size[1]
            flow = flow[y1:y2, x1:x2]
        if padding is not None:
            py, px = padding
            flow = numpy.pad(flow, pad_width=((py, py), (px, px), (0, 0)), mode='constant', constant_values=0)
        if flip:
            flow = numpy.flip(flow, axis=1)
        cf_flow = numpy.moveaxis(flow.astype('float32'), [0, 1, 2], [1, 2, 0])
        return cf_flow

    def load_test_data(self, video_name: str, seq_name: int, pred_frame_num: int):
        data_dict = self.load_testing_data(video_name, seq_name, pred_frame_num, padding=(6, 0))

        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key][None]
            elif isinstance(data_dict[key], list):
                for i in range(len(data_dict[key])):
                    data_dict[key][i] = data_dict[key][i][None]
        return data_dict

    @staticmethod
    def post_process_tensor(tensor: torch.Tensor):
        if tensor.ndim == 4:
            processed_tensor = numpy.moveaxis(tensor.detach().cpu().numpy(), [0, 1, 2, 3], [0, 3, 1, 2])[0, 6:-6]
        elif tensor.ndim == 5:
            processed_tensor = numpy.moveaxis(tensor.detach().cpu().numpy(), [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])[0, 6:-6]
        return processed_tensor
