# Shree KRISHNAya Namaha
# Extracts data from unzipped files into a format suitable for me
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import time
import datetime
import traceback
from enum import Enum
from typing import List

import numpy
import skimage.io

from pathlib import Path
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


# ------------- Enums for easier data passing ---------- #
class DataFeatures(Enum):
    RGB_ALBEDO = 'rgb_albedo'
    RGB_CLEAN = 'rgb_clean'
    RGB_FINAL = 'rgb_final'
    DEPTH = 'depth'
    FLOW = 'flow'
    FLOW_INVALID = 'flow_invalid'
    FLOW_OCCLUSION = 'flow_occlusion'
    INTRINSIC = 'intrinsic'
    TRANSFORMATION = 'extrinsic'


TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'
TRAIN_FEATURES = [DataFeatures.RGB_ALBEDO, DataFeatures.RGB_CLEAN, DataFeatures.RGB_FINAL, DataFeatures.DEPTH,
                  DataFeatures.FLOW, DataFeatures.FLOW_INVALID, DataFeatures.FLOW_OCCLUSION,
                  DataFeatures.INTRINSIC, DataFeatures.TRANSFORMATION]
TEST_FEATURES = [DataFeatures.RGB_CLEAN, DataFeatures.RGB_FINAL]


class DataExtractor:
    def __init__(self, unzipped_dirpath: Path, extract_dirpath: Path, features: List[DataFeatures],
                 train_filepath: Path, test_filepath: Path):
        self.unzipped_dirpath = unzipped_dirpath
        self.extract_dirpath = extract_dirpath
        self.features = features
        self.train_video_names = self.read_video_names(train_filepath)
        self.test_video_names = self.read_video_names(test_filepath)
        self.train_features = list(filter(lambda x: x in TRAIN_FEATURES, features))
        self.test_features = list(filter(lambda x: x in TEST_FEATURES, features))
        return

    @staticmethod
    def read_video_names(path: Path):
        with open(path.as_posix(), 'r') as f:
            video_names = [name.strip() for name in f.readlines()]
        return video_names

    def extract_data(self):
        # Extract train files
        train_rgb_dirpath = self.unzipped_dirpath / 'MPI-Sintel-training_images/training'
        train_depth_dirpath = self.unzipped_dirpath / 'MPI-Sintel-depth-training-20150305/training/depth'
        train_flow_dirpath = self.unzipped_dirpath / 'MPI-Sintel-training_extras/training'
        train_cam_dirpath = self.unzipped_dirpath / 'MPI-Sintel-depth-training-20150305/training/camdata_left'
        output_dirpath = self.extract_dirpath / 'all'
        for video_name in self.train_video_names:
            self.extract_video_data(video_name, self.train_features, train_rgb_dirpath, train_depth_dirpath,
                                    train_flow_dirpath, train_cam_dirpath, output_dirpath)
        print('Extracting train data complete')

        # Extract test files
        test_rgb_dirpath = self.unzipped_dirpath / 'MPI-Sintel-testing'
        return

    def extract_video_data(self, video_name: str, features: List[DataFeatures], rgb_dirpath: Path, depth_dirpath: Path,
                           flow_dirpath: Path, cam_dirpath: Path, output_dirpath: Path):
        rgb_albedo_dirpath = rgb_dirpath / 'albedo'
        rgb_clean_dirpath = rgb_dirpath / 'clean'
        rgb_final_dirpath = rgb_dirpath / 'final'
        num_samples = len(list((rgb_albedo_dirpath / video_name).iterdir()))
        rgb_albedo_list, rgb_clean_list, rgb_final_list, depth_list = [], [], [], []
        flow_list, flow_invalid_list, flow_occlusion_list = [], [], []
        transformation_list, intrinsic_list = [], []
        for i in tqdm(range(num_samples), desc=video_name):
            frame_num = i + 1
            if DataFeatures.RGB_ALBEDO in features:
                frame_path = rgb_albedo_dirpath / f'{video_name}/frame_{frame_num:04}.png'
                frame = self.read_image(frame_path)
                rgb_albedo_list.append(frame)

            if DataFeatures.RGB_CLEAN in features:
                frame_path = rgb_clean_dirpath / f'{video_name}/frame_{frame_num:04}.png'
                frame = self.read_image(frame_path)
                rgb_clean_list.append(frame)

            if DataFeatures.RGB_FINAL in features:
                frame_path = rgb_final_dirpath / f'{video_name}/frame_{frame_num:04}.png'
                frame = self.read_image(frame_path)
                rgb_final_list.append(frame)

            if DataFeatures.DEPTH in features:
                depth_path = depth_dirpath / f'{video_name}/frame_{frame_num:04}.dpt'
                depth = self.read_depth(depth_path)
                depth_list.append(depth)

            if DataFeatures.FLOW in features:
                raise NotImplementedError(f'{DataFeatures.FLOW} not currently supported')

            if DataFeatures.FLOW_INVALID in features:
                raise NotImplementedError(f'{DataFeatures.FLOW_INVALID} not currently supported')

            if DataFeatures.FLOW_OCCLUSION in features:
                raise NotImplementedError(f'{DataFeatures.FLOW_OCCLUSION} not currently supported')

            if (DataFeatures.INTRINSIC in features) or (DataFeatures.TRANSFORMATION in features):
                cam_path = cam_dirpath / f'{video_name}/frame_{frame_num:04}.cam'
                intrinsic, extrinsic = self.read_camera_params(cam_path)
                transformation = numpy.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)
                transformation_list.append(transformation.ravel())
                intrinsic_list.append(intrinsic.ravel())

        if DataFeatures.RGB_ALBEDO in features:
            self.save_frames(output_dirpath / f'{video_name}/rgb/albedo', video_name, rgb_albedo_list)
        if DataFeatures.RGB_CLEAN in features:
            self.save_frames(output_dirpath / f'{video_name}/rgb/clean', video_name, rgb_clean_list)
        if DataFeatures.RGB_FINAL in features:
            self.save_frames(output_dirpath / f'{video_name}/rgb/final', video_name, rgb_final_list)
        if DataFeatures.DEPTH in features:
            self.save_depths(output_dirpath / f'{video_name}/depth', video_name, depth_list)
        if DataFeatures.TRANSFORMATION in features:
            self.save_transformations(output_dirpath / f'{video_name}/TransformationMatrices.csv', transformation_list)
        if DataFeatures.INTRINSIC in features:
            self.save_intrinsics(output_dirpath / f'{video_name}/CameraIntrinsics.csv', intrinsic_list)
        print(f'Extracted video: {video_name}')
        return

    @staticmethod
    def read_image(path: Path):
        image = skimage.io.imread(path.as_posix())
        return image
    
    @staticmethod
    def read_depth(path: Path):
        with open(path.as_posix(), 'rb') as f:
            tag = numpy.fromfile(f, dtype=numpy.float32, count=1)[0]
            assert tag == TAG_FLOAT, f'Incorrect tag: {tag}'
            width = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]
            height = numpy.fromfile(f, dtype=numpy.int32, count=1)[0]
            size = width * height
            assert width > 0 and height > 0 and 1 < size < 100000000
            depth = numpy.fromfile(f, dtype=numpy.float32, count=-1).reshape((height, width))
        return depth
    
    @staticmethod
    def read_camera_params(path: Path):
        with open(path.as_posix(), 'rb') as f:
            tag = numpy.fromfile(f, dtype=numpy.float32, count=1)[0]
            assert tag == TAG_FLOAT, f'Incorrect tag: {tag}'
            intrinsic = numpy.fromfile(f, dtype='float64', count=9).reshape((3, 3))
            extrinsic = numpy.fromfile(f, dtype='float64', count=12).reshape((3, 4))
        return intrinsic, extrinsic

    @staticmethod
    def save_frames(output_dirpath: Path, video_name: str, frames: List[numpy.ndarray]):
        output_dirpath.mkdir(parents=True, exist_ok=True)
        for frame_num, frame in enumerate(tqdm(frames, desc=f'{video_name} frames')):
            output_path = output_dirpath / f'{frame_num:04}.png'
            skimage.io.imsave(output_path.as_posix(), frame)
        return

    @staticmethod
    def save_depths(output_dirpath: Path, video_name: str, depths: List[numpy.ndarray]):
        output_dirpath.mkdir(parents=True, exist_ok=True)
        for frame_num, depth in enumerate(tqdm(depths, desc=f'{video_name} depths')):
            output_path = output_dirpath / f'{frame_num:04}.npz'
            numpy.savez_compressed(output_path.as_posix(), depth=depth)
        return

    @staticmethod
    def save_transformations(output_path: Path, transformations: List[numpy.ndarray]):
        transformations = numpy.stack(transformations, axis=0)
        # noinspection PyTypeChecker
        numpy.savetxt(output_path.as_posix(), transformations, delimiter=',')
        return

    @staticmethod
    def save_intrinsics(output_path: Path, intrinsics: List[numpy.ndarray]):
        intrinsics = numpy.stack(intrinsics, axis=0)
        # noinspection PyTypeChecker
        numpy.savetxt(output_path.as_posix(), intrinsics, delimiter=',')
        return


def demo1():
    root_dirpath = Path('../')
    unzipped_data_dirpath = root_dirpath / 'Data/UnzippedData'
    extracted_data_dirpath = root_dirpath / 'Data/ExtractedData'
    train_video_names_path = root_dirpath / 'res/TrainVideoNames.txt'
    test_video_names_path = root_dirpath / 'res/TestVideoNames.txt'
    features = [DataFeatures.RGB_CLEAN, DataFeatures.DEPTH, DataFeatures.TRANSFORMATION, DataFeatures.INTRINSIC]

    extractor = DataExtractor(unzipped_data_dirpath, extracted_data_dirpath, features, train_video_names_path,
                              test_video_names_path)
    extractor.extract_data()
    return


def main():
    demo1()
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
        run_result = str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
