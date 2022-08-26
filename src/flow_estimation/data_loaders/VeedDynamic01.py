# Shree KRISHNAYa Namaha
# DataLoader for IISc VEED-Dynamic database.
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import random
from pathlib import Path
from typing import Optional

import numpy
import pandas
import torch

from flow_estimation.data_loaders.DataLoaderParent import DataLoaderParent


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
        self.size = configs['flow_estimation']['data_loader']['patch_size']
        self.num_mpi_planes = configs['flow_estimation']['data_loader']['num_mpi_planes']
        if 'resolution_suffix' in self.configs['flow_estimation']['data_loader']:
            self.resolution_suffix = self.configs['flow_estimation']['data_loader']['resolution_suffix']
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

    def load_training_data(self, index: int, random_crop: bool = True, random_flip: bool = True):
        video_name, seq_num, pred_frame_num = self.video_names[index]
        frame1_num = pred_frame_num - 1
        frame1_path = self.dataroot / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/rgb{self.resolution_suffix}/{frame1_num:04}.npy'
        depth1_path = self.dataroot / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/depth{self.resolution_suffix}/{frame1_num:04}.npy'

        if random.randint(0, 1) == 0:
            pw_dirname_f2 = '2step_forward'
            pw_dirname_f0 = '1step_backward'
            poi_dirname = 'LOF_POI_01_2step_backward'
            num_steps12 = -2
            num_steps10 = 1
        else:
            pw_dirname_f2 = '1step_backward'
            pw_dirname_f0 = '2step_forward'
            poi_dirname = 'LOF_POI_02_1step_forward'
            num_steps12 = 1
            num_steps10 = -2

        frame2_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname_f2}/' \
                                      f'warped_frames{self.resolution_suffix}/{frame1_num:04}.npy'
        mask2_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname_f2}/' \
                                     f'masks{self.resolution_suffix}/{frame1_num:04}.npy'
        depth2_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname_f2}/' \
                                      f'warped_depths{self.resolution_suffix}/{frame1_num:04}.npy'
        frame0_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname_f0}/' \
                                      f'warped_frames{self.resolution_suffix}/{frame1_num:04}.npy'
        mask0_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname_f0}/' \
                                     f'masks{self.resolution_suffix}/{frame1_num:04}.npy'
        depth0_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname_f0}/' \
                                      f'warped_depths{self.resolution_suffix}/{frame1_num:04}.npy'
        poi_mask1_path = self.dataroot / f'all_short/LOF_POI/{poi_dirname}/{video_name}/seq{seq_num:02}/' \
                                         f'{frame1_num:04}{self.resolution_suffix}.npy'

        frame0 = self.read_npy_file(frame0_path, mmap_mode='r')
        frame1 = self.read_npy_file(frame1_path, mmap_mode='r')
        frame2 = self.read_npy_file(frame2_path, mmap_mode='r')
        mask0 = self.read_npy_file(mask0_path, mmap_mode='r')
        mask2 = self.read_npy_file(mask2_path, mmap_mode='r')
        depth0 = self.read_npy_file(depth0_path, mmap_mode='r')
        depth1 = self.read_npy_file(depth1_path, mmap_mode='r')
        depth2 = self.read_npy_file(depth2_path, mmap_mode='r')
        poi_mask1 = self.read_npy_file(poi_mask1_path, mmap_mode='r') if poi_mask1_path.exists() else None

        if random_crop:
            psp = self.select_patch_start_point(poi_mask1)
        else:
            psp = None

        if random_flip:
            flip = random.randint(0, 1) == 0
        else:
            flip = False

        padding = None

        frame0 = self.preprocess_frame(frame0, psp, padding, flip)
        frame1 = self.preprocess_frame(frame1, psp, padding, flip)
        frame2 = self.preprocess_frame(frame2, psp, padding, flip)
        mask0 = self.preprocess_mask(mask0, psp, padding, flip)
        mask1 = numpy.ones_like(mask0)
        mask2 = self.preprocess_mask(mask2, psp, padding, flip)
        depth0 = self.preprocess_depth(depth0, psp, padding, flip)
        depth1 = self.preprocess_depth(depth1, psp, padding, flip)
        depth2 = self.preprocess_depth(depth2, psp, padding, flip)

        min_depth = min((depth0 + (1 - mask0) * 1000).min(), depth1.min(), (depth2 + (1 - mask2) * 1000).min())
        max_depth = max(min(depth0.max(), 1000), min(depth1.max(), 1000), min(depth2.max(), 1000))
        depth_bounds = min_depth, max_depth
        mpi0_rgb, mpi0_alpha, mpi0_depth = self.create_mpi(frame0, mask0, depth0, depth_bounds)[:3]
        mpi1_rgb, mpi1_alpha, mpi1_depth = self.create_mpi(frame1, mask1, depth1, depth_bounds)[:3]
        mpi2_rgb, mpi2_alpha, mpi2_depth = self.create_mpi(frame2, mask2, depth2, depth_bounds)[:3]

        data_dict = {
            'frame0': torch.from_numpy(frame0),
            'frame1': torch.from_numpy(frame1),
            'frame2': torch.from_numpy(frame2),
            'mask0': torch.from_numpy(mask0),
            'mask1': torch.from_numpy(mask1),
            'mask2': torch.from_numpy(mask2),
            'depth0': torch.from_numpy(depth0),
            'depth1': torch.from_numpy(depth1),
            'depth2': torch.from_numpy(depth2),
            'mpi0_rgb': torch.from_numpy(mpi0_rgb),
            'mpi0_alpha': torch.from_numpy(mpi0_alpha),
            'mpi0_depth': torch.from_numpy(mpi0_depth),
            'mpi1_rgb': torch.from_numpy(mpi1_rgb),
            'mpi1_alpha': torch.from_numpy(mpi1_alpha),
            'mpi1_depth': torch.from_numpy(mpi1_depth),
            'mpi2_rgb': torch.from_numpy(mpi2_rgb),
            'mpi2_alpha': torch.from_numpy(mpi2_alpha),
            'mpi2_depth': torch.from_numpy(mpi2_depth),
            'num_steps12': num_steps12,
            'num_steps10': num_steps10,
        }
        return data_dict

    def load_testing_data(self, video_name: str, seq_num: int, pred_frame_num: int, pw_dirname: str,
                          padding: tuple = None):
        psp = None
        flip = False

        frame1_num = pred_frame_num - 1
        frame1_path = self.dataroot / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/rgb{self.resolution_suffix}/{frame1_num:04}.npy'
        depth1_path = self.dataroot / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/depth{self.resolution_suffix}/{frame1_num:04}.npy'
        frame2_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname}/' \
                                      f'warped_frames{self.resolution_suffix}/{frame1_num:04}.npy'
        mask2_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname}/' \
                                     f'masks{self.resolution_suffix}/{frame1_num:04}.npy'
        depth2_path = self.dataroot / f'all_short/PoseWarping/PoseWarping01/{video_name}/seq{seq_num:02}/{pw_dirname}/' \
                                      f'warped_depths{self.resolution_suffix}/{frame1_num:04}.npy'

        frame1 = self.read_npy_file(frame1_path)
        frame2 = self.read_npy_file(frame2_path)
        mask2 = self.read_npy_file(mask2_path)
        mask1 = numpy.ones_like(mask2)
        depth1 = self.read_npy_file(depth1_path)
        depth2 = self.read_npy_file(depth2_path)

        min_depth = min(depth1.min(), (depth2 + (1 - mask2) * 1000).min())
        max_depth = max(min(depth1.max(), 1000), min(depth2.max(), 1000))
        depth_bounds = min_depth, max_depth

        frame1 = self.preprocess_frame(frame1, psp, padding, flip)
        frame2 = self.preprocess_frame(frame2, psp, padding, flip)
        mask1 = self.preprocess_mask(mask1, psp, padding, flip)
        mask2 = self.preprocess_mask(mask2, psp, padding, flip)
        depth1 = self.preprocess_depth(depth1, psp, padding, flip)
        depth2 = self.preprocess_depth(depth2, psp, padding, flip)

        mpi1_rgb, mpi1_alpha, mpi1_depth, mpi1_depth_planes = self.create_mpi(frame1, mask1, depth1, depth_bounds)
        mpi2_rgb, mpi2_alpha, mpi2_depth, mpi2_depth_planes = self.create_mpi(frame2, mask2, depth2, depth_bounds)

        data_dict = {
            'frame1': torch.from_numpy(frame1),
            'frame2': torch.from_numpy(frame2),
            'mask1': torch.from_numpy(mask1),
            'mask2': torch.from_numpy(mask2),
            'depth1': torch.from_numpy(depth1),
            'depth2': torch.from_numpy(depth2),
            'mpi1_rgb': torch.from_numpy(mpi1_rgb),
            'mpi1_alpha': torch.from_numpy(mpi1_alpha),
            'mpi1_depth': torch.from_numpy(mpi1_depth),
            'mpi1_depth_planes': torch.from_numpy(mpi1_depth_planes),
            'mpi2_rgb': torch.from_numpy(mpi2_rgb),
            'mpi2_alpha': torch.from_numpy(mpi2_alpha),
            'mpi2_depth': torch.from_numpy(mpi2_depth),
            'mpi2_depth_planes': torch.from_numpy(mpi2_depth_planes),
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
        cf_depth = depth[None].astype('float32')
        return cf_depth

    def create_mpi(self, frame: numpy.ndarray, mask: numpy.ndarray, depth: numpy.ndarray, depth_bounds):
        depth_planes = self.get_depth_planes(*depth_bounds)
        nearest_plane_index = numpy.argmin(numpy.abs(depth_planes[None, None, None] - depth[:, :, :, None]), axis=3)
        h, w = frame.shape[1:3]
        mpi_alpha = numpy.zeros(shape=(1, h, w, self.num_mpi_planes), dtype=numpy.float32)
        mpi_alpha[numpy.arange(1)[:, None, None], numpy.arange(h)[None, :, None], numpy.arange(w)[None, None, :], nearest_plane_index] = 1
        mpi_alpha[~(numpy.broadcast_to(mask[:, :, :, None], mpi_alpha.shape).astype('bool'))] = 0
        mpi_rgb = frame[:, :, :, None] * mpi_alpha
        mpi_depth = depth[:, :, :, None] * mpi_alpha
        depth_planes = (numpy.broadcast_to(depth_planes[None, None, None], mpi_depth.shape)).copy()
        return mpi_rgb, mpi_alpha, mpi_depth, depth_planes

    def get_depth_planes(self, min_depth, max_depth):
        min_disp = 1 / (min_depth + 1e-3)
        max_disp = 1 / (max_depth + 1e-3)
        disp_planes = numpy.linspace(min_disp, max_disp, self.num_mpi_planes)
        depth_planes = 1 / disp_planes
        return depth_planes

    def load_test_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        data_dict = self.load_testing_data(video_name, seq_num, pred_frame_num, pw_dirname='1step_backward',
                                           padding=(4, 0))

        for key in data_dict.keys():
            data_dict[key] = data_dict[key][None]
        return data_dict

    def load_prediction_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        data_dict = self.load_testing_data(video_name, seq_num, pred_frame_num, pw_dirname='2step_forward',
                                           padding=(4, 0))

        for key in data_dict.keys():
            data_dict[key] = data_dict[key][None]
        return data_dict

    def load_generation_data(self, video_name: str, seq_num: int, frame1_num: int, pw_dirname):
        data_dict = self.load_testing_data(video_name, seq_num, pred_frame_num=frame1_num + 1, pw_dirname=pw_dirname,
                                           padding=(4, 0))

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
