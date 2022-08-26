# Shree KRISHNAYa Namaha
# DataLoader for IISc VEED-Dynamic database
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from pathlib import Path
from typing import Optional

import numpy
import pandas
import torch

from video_inpainting.data_loaders.DataLoaderParent import DataLoaderParent
from utils.MpiUtils import get_depth_planes


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
        self.size = configs['video_inpainting']['data_loader']['patch_size']
        self.lg_dirname = self.configs['video_inpainting']['data_loader']['local_global_dirname']
        self.num_mpi_planes = configs['video_inpainting']['data_loader']['num_mpi_planes']
        if 'resolution_suffix' in self.configs['video_inpainting']['data_loader']:
            self.resolution_suffix = self.configs['video_inpainting']['data_loader']['resolution_suffix']
        else:
            self.resolution_suffix = ''
        self.frame_resolution = self.get_frame_resolution(self.resolution_suffix)
        return

    @staticmethod
    def get_frame_resolution(resolution_suffix):
        h, w = 1080, 1920
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

    def load_training_data(self, index: int, random_crop: bool = True):
        video_name, seq_num, pred_frame_num = self.video_names[index]
        frame_path = self.dataroot / f'all_short/LocalGlobalMotionPredicted/{self.lg_dirname}/{video_name}/seq{seq_num:02}/WarpedFrames{self.resolution_suffix}/{pred_frame_num:04}.npy'
        depth_path = self.dataroot / f'all_short/LocalGlobalMotionPredicted/{self.lg_dirname}/{video_name}/seq{seq_num:02}/WarpedDepths{self.resolution_suffix}/{pred_frame_num:04}.npy'
        mask_path = self.dataroot / f'all_short/LocalGlobalMotionPredicted/{self.lg_dirname}/{video_name}/seq{seq_num:02}/Masks{self.resolution_suffix}/{pred_frame_num:04}.npy'
        target_frame_path = self.dataroot / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/rgb{self.resolution_suffix}/{pred_frame_num:04}.npy'

        frame = self.read_npy_file(frame_path, mmap_mode='r')
        depth = self.read_npy_file(depth_path, mmap_mode='r')
        mask = self.read_npy_file(mask_path, mmap_mode='r')
        target_frame = self.read_npy_file(target_frame_path, mmap_mode='r')

        if random_crop:
            psp = self.select_patch_start_point(mask)
        else:
            psp = None

        flip = False
        padding = None

        frame = self.preprocess_frame(frame, psp, padding, flip)
        depth = self.preprocess_depth(depth, psp, padding, flip)
        mask = self.preprocess_mask(mask, psp, padding, flip)
        target_frame = self.preprocess_frame(target_frame, psp, padding, flip)

        min_depth = (depth + 1000 * (1 - mask)).min()
        max_depth = min(depth.max(), 1000)
        depth_bounds = min_depth, max_depth
        mpi_rgb, mpi_alpha, mpi_depth = self.create_mpi(frame, mask, depth, depth_bounds)

        data_dict = {
            'frame': torch.from_numpy(frame),
            'mask': torch.from_numpy(mask),
            'depth': torch.from_numpy(depth),
            'target_frame': torch.from_numpy(target_frame),
            'mpi_rgb': torch.from_numpy(mpi_rgb),
            'mpi_alpha': torch.from_numpy(mpi_alpha),
            'mpi_depth': torch.from_numpy(mpi_depth),
        }
        return data_dict

    def load_testing_data(self, video_name: str, seq_num: int, pred_frame_num: int, padding: tuple = None):
        psp = None
        flip = False

        frame_path = self.dataroot / f'all_short/LocalGlobalMotionPredicted/{self.lg_dirname}/{video_name}/seq{seq_num:02}/WarpedFrames{self.resolution_suffix}/{pred_frame_num:04}.npy'
        depth_path = self.dataroot / f'all_short/LocalGlobalMotionPredicted/{self.lg_dirname}/{video_name}/seq{seq_num:02}/WarpedDepths{self.resolution_suffix}/{pred_frame_num:04}.npy'
        mask_path = self.dataroot / f'all_short/LocalGlobalMotionPredicted/{self.lg_dirname}/{video_name}/seq{seq_num:02}/Masks{self.resolution_suffix}/{pred_frame_num:04}.npy'

        frame = self.read_npy_file(frame_path)
        depth = self.read_npy_file(depth_path)
        mask = self.read_npy_file(mask_path)

        frame = self.preprocess_frame(frame, psp, padding, flip)
        depth = self.preprocess_depth(depth, psp, padding, flip)
        mask = self.preprocess_mask(mask, psp, padding, flip)

        min_depth = (depth + 1000 * (1 - mask)).min()
        max_depth = min(depth.max(), 1000)
        depth_bounds = min_depth, max_depth
        mpi_rgb, mpi_alpha, mpi_depth = self.create_mpi(frame, mask, depth, depth_bounds)

        data_dict = {
            'frame': torch.from_numpy(frame),
            'mask': torch.from_numpy(mask),
            'depth': torch.from_numpy(depth),
            'mpi_rgb': torch.from_numpy(mpi_rgb),
            'mpi_alpha': torch.from_numpy(mpi_alpha),
            'mpi_depth': torch.from_numpy(mpi_depth),
        }
        return data_dict

    def select_patch_start_point(self, mask: numpy.ndarray):
        # Crop mask such that a patch centered around any point in cropped mask doesn't go out of bounds
        a, b = self.size[0] // 2, self.size[1] // 2
        h, w = self.frame_resolution
        start_point = None
        if mask is not None:
            mask_cropped = mask[a:-a, b:-b]
            interest_points = numpy.argwhere(mask_cropped == 1)
            if not((interest_points.ndim != 2) or (interest_points.shape[0] == 0)):
                start_point = interest_points[numpy.random.randint(0, interest_points.shape[0])]
        if start_point is None:
            start_point = [h // 2 - a, w // 2 - b]
        return start_point

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
        depth = numpy.clip(depth, a_min=0, a_max=1000)
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

    def create_mpi(self, frame: numpy.ndarray, mask: numpy.ndarray, depth: numpy.ndarray, depth_bounds):
        depth_planes = get_depth_planes(self.num_mpi_planes, *depth_bounds)
        nearest_plane_index = numpy.argmin(numpy.abs(depth_planes[None, None, None] - depth[:, :, :, None]), axis=3)
        h, w = frame.shape[1:3]
        mpi_alpha = numpy.zeros(shape=(1, h, w, self.num_mpi_planes), dtype=numpy.float32)
        mpi_alpha[numpy.arange(1)[:, None, None], numpy.arange(h)[None, :, None], numpy.arange(w)[None, None, :], nearest_plane_index] = 1
        mpi_alpha[~(numpy.broadcast_to(mask[:, :, :, None], mpi_alpha.shape).astype('bool'))] = 0
        mpi_rgb = frame[:, :, :, None] * mpi_alpha
        mpi_depth = depth[:, :, :, None] * mpi_alpha
        depth_planes = depth_planes[None, None, None]
        return mpi_rgb, mpi_alpha, mpi_depth

    def load_test_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        data_dict = self.load_testing_data(video_name, seq_num, pred_frame_num, padding=(4, 0))

        for key in data_dict.keys():
            data_dict[key] = data_dict[key][None]
        return data_dict

    @staticmethod
    def post_process_tensor(tensor: torch.Tensor):
        if tensor.ndim == 4:
            processed_tensor = numpy.moveaxis(tensor.detach().cpu().numpy(), [0, 1, 2, 3], [0, 3, 1, 2])[0, 4:-4]
        elif tensor.ndim == 5:
            processed_tensor = numpy.moveaxis(tensor.detach().cpu().numpy(), [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])[0, 4:-4]
        return processed_tensor
