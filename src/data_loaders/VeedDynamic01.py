# Shree KRISHNAYa Namaha
# DataLoader for IISc VEED-Dynamic dataset
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
        if self.configs['data_loader']['video_duration'] == 'short':
            self.all_dirname = 'all_short'
        else:
            self.all_dirname = 'all_long'
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

    def load_testing_data(self, video_name: str, seq_num: int, pred_frame_num: int, padding: tuple = None):
        psp = None
        flip = False

        frame1_num = pred_frame_num - 1
        frame0_num = frame1_num - self.num_steps
        frame0_path = self.dataroot / f'{self.all_dirname}/RenderedData/{video_name}/seq{seq_num:02}/rgb{self.resolution_suffix}/{frame0_num:04}.png'
        frame1_path = self.dataroot / f'{self.all_dirname}/RenderedData/{video_name}/seq{seq_num:02}/rgb{self.resolution_suffix}/{frame1_num:04}.png'
        depth0_path = self.dataroot / f'{self.all_dirname}/RenderedData/{video_name}/seq{seq_num:02}/depth{self.resolution_suffix}/{frame0_num:04}.exr'
        depth1_path = self.dataroot / f'{self.all_dirname}/RenderedData/{video_name}/seq{seq_num:02}/depth{self.resolution_suffix}/{frame1_num:04}.exr'
        transformation_path = self.dataroot / f'{self.all_dirname}/RenderedData/{video_name}/seq{seq_num:02}/' \
                                              f'TransformationMatrices.csv'

        frame0 = self.read_image(frame0_path)
        frame1 = self.read_image(frame1_path)
        depth0 = self.read_depth(depth0_path)
        depth1 = self.read_depth(depth1_path)
        transformation_matrices = numpy.genfromtxt(transformation_path, delimiter=',', dtype=numpy.float32).reshape((-1, 4, 4))
        transformation0 = transformation_matrices[frame0_num]
        transformation1 = transformation_matrices[frame1_num]
        transformation2 = [transformation_matrices[pred_frame_num + i] for i in range(self.num_pred_frames)]
        intrinsic = self.get_camera_intrinsic_transform(patch_start_point=psp, padding=padding)

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
            'intrinsic0': torch.from_numpy(intrinsic),
            'intrinsic1': torch.from_numpy(intrinsic),
            'intrinsic2': [torch.from_numpy(intrinsic) for _ in range(self.num_pred_frames)],
        }

        if self.configs['data_loader']['load_true_flow']:
            flow_path = self.dataroot / f'{self.all_dirname}/RenderedData/{video_name}/seq{seq_num:02}/optical_flow{self.resolution_suffix}/{frame1_num:04}.exr'
            flow12 = self.read_flow(flow_path)
            flow12 = self.preprocess_flow(flow12, psp, padding, flip)
            data_dict['flow12'] = torch.from_numpy(flow12)

        if self.configs['data_loader']['load_target_frames']:
            target_frame_paths = [self.dataroot / f'{self.all_dirname}/RenderedData/{video_name}/seq{seq_num:02}/rgb{self.resolution_suffix}/{pred_frame_num + i:04}.npy' for i in range(self.num_pred_frames)]
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
    def read_image(path: Path):
        if path.suffix == '.png':
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def read_depth(path: Path):
        if path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.exr':
            import OpenEXR, Imath

            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth

    @staticmethod
    def read_flow(path: Path):
        import OpenEXR, Imath

        exr_file = OpenEXR.InputFile(path.as_posix())
        height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
        width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x

        raw_bytes_r = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
        vector_r = numpy.frombuffer(raw_bytes_r, dtype=numpy.float32)
        of_r = numpy.reshape(vector_r, (height, width))

        raw_bytes_g = exr_file.channel('G', Imath.PixelType(Imath.PixelType.FLOAT))
        vector_g = numpy.frombuffer(raw_bytes_g, dtype=numpy.float32)
        of_g = numpy.reshape(vector_g, (height, width))

        raw_bytes_b = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
        vector_b = numpy.frombuffer(raw_bytes_b, dtype=numpy.float32)
        of_b = numpy.reshape(vector_b, (height, width))

        raw_bytes_a = exr_file.channel('A', Imath.PixelType(Imath.PixelType.FLOAT))
        vector_a = numpy.frombuffer(raw_bytes_a, dtype=numpy.float32)
        of_a = numpy.reshape(vector_a, (height, width))

        backward_flow = numpy.stack([of_r, -of_g], axis=2)
        forward_flow = numpy.stack([-of_b, of_a], axis=2)
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

    @staticmethod
    def get_camera_intrinsic_transform(capture_width=1920, capture_height=1080,
                                       patch_start_point: Optional[tuple] = (0, 0), padding: Optional[tuple] = (0, 0)):
        if patch_start_point is None:
            patch_start_point = (0, 0)
        if padding is None:
            padding = (0, 0)
        start_y, start_x = patch_start_point
        pad_y, pad_x = padding
        camera_intrinsics = numpy.eye(3, dtype=numpy.float32)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x + pad_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y + pad_y
        return camera_intrinsics

    def load_test_data(self, video_name: str, seq_num: int, pred_frame_num: int):
        data_dict = self.load_testing_data(video_name, seq_num, pred_frame_num, padding=(4, 0))

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
            processed_tensor = numpy.moveaxis(tensor.detach().cpu().numpy(), [0, 1, 2, 3], [0, 3, 1, 2])[0, 4:-4]
        elif tensor.ndim == 5:
            processed_tensor = numpy.moveaxis(tensor.detach().cpu().numpy(), [0, 1, 2, 3, 4], [0, 4, 1, 2, 3])[0, 4:-4]
        return processed_tensor
