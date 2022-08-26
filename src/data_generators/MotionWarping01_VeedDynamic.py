# Shree KRISHNAya Namaha
# Total forward flow (local + global) is computed.
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import inspect
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional, List

import numpy
import pandas
import simplejson
import skimage.io
from deepdiff import DeepDiff
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class Warper:
    def __init__(self, resolution: tuple):
        self.resolution = resolution
        self.grid = self.create_grid(*self.resolution)
        return

    def compute_total_forward_flow(self, local_flow10: numpy.ndarray, depth1: numpy.ndarray,
                                   transformation1: numpy.ndarray, transformation2: numpy.ndarray,
                                   intrinsic1: numpy.ndarray, intrinsic2: numpy.ndarray):
        local_flow12 = -local_flow10
        local_flow12xy = local_flow12[:, :, :2]
        local_flow12z = local_flow12[:, :, 2]
        pos_vectors = local_flow12xy + self.grid
        depth1a = local_flow12z + depth1

        trans_points12 = self.compute_transformed_points(depth1a, transformation1, transformation2, intrinsic1, intrinsic2,
                                                         pos_vectors=pos_vectors)
        trans_depth2 = trans_points12[:, :, 2:3, 0]
        trans_coordinates12 = trans_points12[:, :, :2, 0] / trans_depth2
        total_flow12xy = trans_coordinates12 - self.grid
        total_flow12z = trans_depth2 - depth1[:, :, None]
        total_flow12 = numpy.concatenate([total_flow12xy, total_flow12z], axis=2)
        return total_flow12

    def compute_transformed_points(self, depth1: numpy.ndarray, transformation1: numpy.ndarray,
                                   transformation2: numpy.ndarray, intrinsic1: numpy.ndarray, intrinsic2: numpy.ndarray,
                                   *, pos_vectors: Optional[numpy.ndarray] = None):
        """
        Computes transformed position for each pixel location
        """
        h, w = self.resolution
        assert depth1.shape == self.resolution
        transformation = numpy.matmul(transformation2, numpy.linalg.inv(transformation1))

        if pos_vectors is None:
            y1d = numpy.array(range(h))
            x1d = numpy.array(range(w))
            x2d, y2d = numpy.meshgrid(x1d, y1d)
        else:
            x2d = pos_vectors[:, :, 0]  # (h, w)
            y2d = pos_vectors[:, :, 1]  # (h, w)
        ones_2d = numpy.ones(shape=(h, w))  # (h, w)
        ones_4d = ones_2d[:, :, None, None]
        pos_vectors_homo = numpy.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]  # (h, w, 3, 1)

        intrinsic1_inv = numpy.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[None, None]
        intrinsic2_4d = intrinsic2[None, None]
        depth_4d = depth1[:, :, None, None]
        trans_4d = transformation[None, None]

        unnormalized_pos = numpy.matmul(intrinsic1_inv_4d, pos_vectors_homo)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = numpy.concatenate([world_points, ones_4d], axis=2)
        trans_world_homo = numpy.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :3]
        trans_norm_points = numpy.matmul(intrinsic2_4d, trans_world)
        return trans_norm_points

    def bilinear_splatting(self, frame1: numpy.ndarray, mask1: Optional[numpy.ndarray], depth1: numpy.ndarray,
                           flow12: numpy.ndarray, flow12_mask: Optional[numpy.ndarray], is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using inverse bilinear interpolation based splatting
        :param frame1: (h, w, c)
        :param mask1: (h, w): True if known and False if unknown. Optional
        :param depth1: (h, w)
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame2: (h, w, c)
                 mask2: (h, w): True if known and False if unknown
        """
        h, w, c = frame1.shape
        if mask1 is None:
            mask1 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        sat_depth1 = numpy.clip(depth1, a_min=0, a_max=1000)
        log_depth1 = numpy.log(1 + sat_depth1)
        depth_weights = numpy.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
        weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
        weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
        weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        warped_image = numpy.zeros(shape=(h + 2, w + 2, c), dtype=numpy.float64)
        warped_weights = numpy.zeros(shape=(h + 2, w + 2), dtype=numpy.float64)

        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

        cropped_warped_image = warped_image[1:-1, 1:-1]
        cropped_weights = warped_weights[1:-1, 1:-1]

        mask = cropped_weights > 0
        with numpy.errstate(invalid='ignore'):
            warped_frame2 = numpy.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

        if is_image:
            assert numpy.min(warped_frame2) >= 0
            assert numpy.max(warped_frame2) <= 256
            clipped_image = numpy.clip(warped_frame2, a_min=0, a_max=255)
            warped_frame2 = numpy.round(clipped_image).astype('uint8')
        return warped_frame2, mask

    def bilinear_interpolation(self, frame2: numpy.ndarray, mask2: Optional[numpy.ndarray], flow12: numpy.ndarray,
                               flow12_mask: Optional[numpy.ndarray], is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using bilinear interpolation
        :param frame2: (h, w, c)
        :param mask2: (h, w): True if known and False if unknown. Optional
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame1: (h, w, c)
                 mask1: (h, w): True if known and False if unknown
        """
        if self.resolution is not None:
            assert frame2.shape[:2] == self.resolution
        h, w, c = frame2.shape
        if mask2 is None:
            mask2 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, 0] = numpy.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = numpy.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        weight_nw = prox_weight_nw * flow12_mask
        weight_sw = prox_weight_sw * flow12_mask
        weight_ne = prox_weight_ne * flow12_mask
        weight_se = prox_weight_se * flow12_mask

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        frame2_offset = numpy.pad(frame2, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        mask2_offset = numpy.pad(mask2, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

        f2_nw = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_sw = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_ne = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        f2_se = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_sw = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_ne = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        m2_se = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw_3d = m2_nw[:, :, None]
        m2_sw_3d = m2_sw[:, :, None]
        m2_ne_3d = m2_ne[:, :, None]
        m2_se_3d = m2_se[:, :, None]

        nr = weight_nw_3d * f2_nw * m2_nw_3d + weight_sw_3d * f2_sw * m2_sw_3d + \
             weight_ne_3d * f2_ne * m2_ne_3d + weight_se_3d * f2_se * m2_se_3d
        dr = weight_nw_3d * m2_nw_3d + weight_sw_3d * m2_sw_3d + weight_ne_3d * m2_ne_3d + weight_se_3d * m2_se_3d
        warped_frame1 = numpy.where(dr > 0, nr / dr, 0)
        mask1 = dr[:, :, 0] > 0

        if is_image:
            assert numpy.min(warped_frame1) >= 0
            assert numpy.max(warped_frame1) <= 256
            clipped_image = numpy.clip(warped_frame1, a_min=0, a_max=255)
            warped_frame1 = numpy.round(clipped_image).astype('uint8')
        return warped_frame1, mask1

    @staticmethod
    def create_grid(h, w):
        x_1d = numpy.arange(0, w)[None]
        y_1d = numpy.arange(0, h)[:, None]
        x_2d = numpy.repeat(x_1d, repeats=h, axis=0)
        y_2d = numpy.repeat(y_1d, repeats=w, axis=1)
        grid = numpy.stack([x_2d, y_2d], axis=2)
        return grid

    @staticmethod
    def read_image(path: Path):
        if path.suffix in ['.png']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def save_image(path: Path, image: numpy.ndarray, as_png: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in ['.png']:
            skimage.io.imsave(path.as_posix(), image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), image)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), image)
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return

    @staticmethod
    def read_mask(path: Path):
        if path.suffix in ['.png']:
            mask = skimage.io.imread(path.as_posix()) == 255
        elif path.suffix == '.npy':
            mask = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown mask format: {path.as_posix()}')
        return mask

    @staticmethod
    def save_mask(path: Path, mask: numpy.ndarray, as_png: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        mask_image = mask.astype('uint8') * 255
        if path.suffix in ['.jpg', '.png', '.bmp']:
            skimage.io.imsave(path.as_posix(), mask_image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), mask)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), mask_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown format: {path.as_posix()}')
        return

    def read_depth(self, path: Path) -> numpy.ndarray:
        if path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        elif path.suffix == '.exr':
            import Imath
            import OpenEXR

            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height, width = self.resolution
            # height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            # width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth

    @staticmethod
    def save_depth(path: Path, depth: numpy.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.npy':
            numpy.save(path.as_posix(), depth)
        elif path.suffix == '.npz':
            numpy.savez_compressed(path, depth=depth)
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return

    @staticmethod
    def read_transformed_points(path: Path):
        if path.suffix == '.npy':
            transformed_points = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as tp_data:
                transformed_points = tp_data['transformed_points']
        else:
            raise RuntimeError(f'Unknown transformed points format: {path.as_posix()}')
        return transformed_points

    @staticmethod
    def save_transformed_points(path: Path, transformed_points: numpy.ndarray):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.npy':
            numpy.save(path.as_posix(), transformed_points)
        elif path.suffix == '.npz':
            numpy.savez_compressed(path, transformed_points=transformed_points)
        else:
            raise RuntimeError(f'Unknown transformed points format: {path.as_posix()}')
        return

    @staticmethod
    def read_flow(path: Path):
        if path.suffix == '.npy':
            flow = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as flow_data:
                flow = flow_data['flow']
        else:
            raise RuntimeError(f'Unknown flow format: {path.as_posix()}')
        return flow

    @staticmethod
    def save_flow(path: Path, flow: numpy.ndarray):
        flow = flow.astype('float32')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.npy':
            numpy.save(path.as_posix(), flow)
        elif path.suffix == '.npz':
            numpy.savez_compressed(path, flow=flow)
        else:
            raise RuntimeError(f'Unknown flow format: {path.as_posix()}')
        return


def camera_intrinsic_transform(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
    """
    Based on Warper.camera_intrinsic_transform_05()
    """
    start_y, start_x = patch_start_point
    camera_intrinsics = numpy.eye(3)
    camera_intrinsics[0, 0] = 2100
    camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
    camera_intrinsics[1, 1] = 2100
    camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
    return camera_intrinsics


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming data generation: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_generation(gen_configs: dict, frames_data: pandas.DataFrame):
    gen_num = gen_configs['gen_num']

    root_dirpath = Path('../../')
    data_dirpath = root_dirpath / 'Data/Databases/VeedDynamic'
    output_dirpath = data_dirpath / f'all_short/LocalGlobalMotionPredicted/LGMP_{gen_num:02}_{gen_configs["description"]}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, gen_configs)

    past_num_steps = gen_configs['past_num_steps']
    local_flow_dirname = gen_configs['local_flow_dirname']

    warper = Warper(resolution=(1080, 1920))

    rendered_dirpath = data_dirpath / 'all_short/RenderedData'
    local_flow_dirpath = data_dirpath / f'all_short/LocalOpticalFlows/{local_flow_dirname}'
    intrinsic = camera_intrinsic_transform()

    for i, frame_data in tqdm(frames_data.iterrows(), total=frames_data.shape[0]):
        video_name, seq_num, frame1_num = frame_data

        transformation_path = rendered_dirpath / f'{video_name}/seq{seq_num:02}/TransformationMatrices.csv'
        transformation_matrices = numpy.genfromtxt(transformation_path, delimiter=',')

        frame2_num = frame1_num + 1
        frame1_path = rendered_dirpath / f'{video_name}/seq{seq_num:02}/rgb/{frame1_num:04}.npy'
        depth1_path = rendered_dirpath / f'{video_name}/seq{seq_num:02}/depth/{frame1_num:04}.npy'
        local_flow10_path = local_flow_dirpath / f'{video_name}/seq{seq_num:02}/flows/{frame1_num:04}.npy'

        warped_frame2_path = output_dirpath / f'{video_name}/seq{seq_num:02}/WarpedFrames/{frame2_num:04}.npy'
        warped_depth2_path = output_dirpath / f'{video_name}/seq{seq_num:02}/WarpedDepths/{frame2_num:04}.npy'
        mask2_path = output_dirpath / f'{video_name}/seq{seq_num:02}/Masks/{frame2_num:04}.npy'
        total_flow12_path = output_dirpath / f'{video_name}/seq{seq_num:02}/TotalFlows/{frame1_num:04}.npy'

        if warped_frame2_path.exists() and mask2_path.exists() and warped_depth2_path.exists() and \
                total_flow12_path.exists():
            continue

        frame1 = warper.read_image(frame1_path)
        depth1 = warper.read_depth(depth1_path)
        depth1 = numpy.clip(depth1, a_min=0, a_max=1000)
        transformation1 = transformation_matrices[frame1_num].reshape(4, 4)
        transformation2 = transformation_matrices[frame2_num].reshape(4, 4)
        local_flow10 = warper.read_flow(local_flow10_path) / past_num_steps

        if total_flow12_path.exists():
            total_flow12 = warper.read_flow(total_flow12_path)
        else:
            total_flow12 = warper.compute_total_forward_flow(local_flow10, depth1, transformation1, transformation2, intrinsic, intrinsic)

        if warped_frame2_path.exists() and mask2_path.exists():
            warped_frame2 = warper.read_image(warped_frame2_path)
            mask2 = warper.read_mask(mask2_path)
        else:
            warped_frame2, mask2 = warper.bilinear_splatting(frame1, None, depth1, total_flow12[:, :, :2], None, is_image=True)

        if warped_depth2_path.exists():
            warped_depth2 = warper.read_depth(warped_depth2_path)
        else:
            warped_depth2 = warper.bilinear_splatting(depth1[:, :, None], None, depth1, total_flow12[:, :, :2], None, is_image=False)[0][:, :, 0]

        if not warped_frame2_path.exists():
            warper.save_image(warped_frame2_path, warped_frame2, as_png=True)

        if not mask2_path.exists():
            warper.save_mask(mask2_path, mask2, as_png=True)

        if not warped_depth2_path.exists():
            warper.save_depth(warped_depth2_path, warped_depth2)

        if not total_flow12_path.exists():
            warper.save_flow(total_flow12_path, total_flow12)
    return


def create_frames_data01(video_names: List[str], seq_nos: List[int], frame1_nos: List[int]) -> pandas.DataFrame:
    frames_data = []
    for video_name in video_names:
        for seq_num in seq_nos:
            for frame1 in frame1_nos:
                frames_data.append([video_name, seq_num, frame1])
    frames_data = pandas.DataFrame(frames_data, columns=['video_name', 'seq_num', 'frame1_num'])
    return frames_data


def wrapper01(gen_num: int, description: str, set_num: int,
              frame1_start: int, frame1_end: int, frame1_step: int, group: str):
    """
    A wrapper that generates frames_data
    :param gen_num:
    :param description:
    :param set_num:
    :param frame1_start:
    :param frame1_end:
    :param frame1_step:
    :param group: One of ['train', 'validation', 'test']
    :return:
    """
    this_method = inspect.currentframe().f_code.co_name
    configs = {
        'DataGenerator': f'{this_filename}/{this_method}',
        'gen_num': gen_num,
        'description': description,
        'gen_set_num': set_num,
        'past_num_steps': 2,
        'local_flow_dirname': 'LOF_01_2step_backward',
        'database_name': 'VeedDynamic',
    }

    frame1_nos = list(range(frame1_start, frame1_end + 1, frame1_step))
    seq_nos = list(range(4))
    group_videos_datapath = Path(f'../../res/TrainTestSets/VeedDynamic/Set{set_num:02}/{group.capitalize()}VideosData.csv')
    group_video_names = numpy.unique(pandas.read_csv(group_videos_datapath)['video_name'])

    frames_data = create_frames_data01(group_video_names, seq_nos, frame1_nos)
    start_generation(configs, frames_data)
    return


def main():
    wrapper01(gen_num=1, description='1step_forward', set_num=1, frame1_start=2, frame1_end=10, frame1_step=1, group='train')
    wrapper01(gen_num=1, description='1step_forward', set_num=1, frame1_start=2, frame1_end=10, frame1_step=1, group='validation')
    wrapper01(gen_num=1, description='1step_forward', set_num=1, frame1_start=6, frame1_end=8, frame1_step=1, group='test')
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    try:
        main()
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
