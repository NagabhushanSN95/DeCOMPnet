# Shree KRISHNAya Namaha
# Tester
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

from ast import literal_eval
from pathlib import Path
from typing import Union, Optional

import Imath
import OpenEXR
import flow_vis
import numpy
import pandas
import simplejson
import skimage.io
import torch
from tqdm import tqdm

from data_loaders.DataLoaderFactory import get_data_loader
from data_loaders.DataLoaderParent import DataLoaderParent
from flow_estimation.feature_extractors.FeatureExtractorFactory import get_feature_extractor
from flow_estimation.flow_estimators.FlowEstimatorFactory import get_flow_estimator as get_local_flow_estimator
from frame_predictors.FramePredictorFactory import get_frame_predictor
from local_flow_predictors.LocalFlowPredictorFactory import get_local_flow_predictor
from utils import CommonUtils
from video_inpainting.flow_predictors.FlowPredictorFactory import get_flow_predictor as get_infilling_flow_predictor

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class Tester:
    data_loader: Optional[DataLoaderParent]

    def __init__(self, root_dirpath: Path, database_dirpath: Path, train_configs: dict, device: str = 'gpu0'):
        self.root_dirpath = root_dirpath
        self.database_dirpath = database_dirpath
        self.train_configs = train_configs
        self.device = CommonUtils.get_device(device)

        self.frame_predictor = None
        self.data_loader = None
        self.build_model()
        self.get_data_loader()
        return

    def build_model(self):
        feature_extractor = None
        local_flow_estimator = None
        local_flow_predictor = None
        infilling_flow_predictor = None
        if 'feature_extractor' in self.train_configs['flow_estimation'].keys():
            feature_extractor = get_feature_extractor(self.train_configs)
        if 'flow_estimator' in self.train_configs['flow_estimation'].keys():
            local_flow_estimator = get_local_flow_estimator(self.train_configs)
        if 'local_flow_predictor' in self.train_configs.keys():
            local_flow_predictor = get_local_flow_predictor(self.train_configs)
        if 'flow_predictor' in self.train_configs['video_inpainting'].keys():
            infilling_flow_predictor = get_infilling_flow_predictor(self.train_configs)
        self.frame_predictor = get_frame_predictor(self.train_configs, feature_extractor, local_flow_estimator, local_flow_predictor, infilling_flow_predictor).to(
            self.device)
        return

    def get_data_loader(self):
        self.data_loader = get_data_loader(self.train_configs, self.database_dirpath, frames_datapath=None)
        return

    def load_model(self, flow_model_path: Path, inpainting_model_path: Path):
        self.frame_predictor.load_weights(flow_model_path, inpainting_model_path)
        return

    def pre_test_ops(self):
        self.frame_predictor.eval()
        return

    def load_data(self, video_name, seq_num, pred_frame_num):
        input_data = self.data_loader.load_test_data(video_name, seq_num, pred_frame_num)
        return input_data

    def predict_next_frame1(self, input_batch: dict, return_intermediate_results):
        input_batch = CommonUtils.move_to_device(input_batch, self.device)
        with torch.no_grad():
            output_batch = self.frame_predictor(input_batch, return_intermediate_results)
        processed_output = self.post_process_output(output_batch)

        pred_frames = [self.post_process_pred_frame(frame) for frame in processed_output['predicted_frames']]
        pred_frames_mask = [self.post_process_mask(mask) for mask in processed_output['predicted_frames_mask']]

        extras = None
        if return_intermediate_results:
            extras = {
                'local_flow12': processed_output['local_flow12'],
                'local_flow_warped_frame2': self.post_process_pred_frame(processed_output['local_flow_warped_frame2']),
                'total_flow_warped_frame2': self.post_process_pred_frame(processed_output['total_flow_warped_frame2'])
            }
        return pred_frames, pred_frames_mask, extras

    def predict_next_frame2(self, video_name, seq_num, pred_frame_num, return_intermediate_results: bool):
        input_dict = self.load_data(video_name, seq_num, pred_frame_num)
        pred_frames, pred_frames_mask, extras = self.predict_next_frame1(input_dict, return_intermediate_results)
        return pred_frames, pred_frames_mask, extras

    def post_process_output(self, output_batch: Union[torch.Tensor, list, tuple, dict]):
        if isinstance(output_batch, torch.Tensor):
            processed_batch = self.data_loader.post_process_tensor(output_batch)
        elif isinstance(output_batch, list) or isinstance(output_batch, tuple):
            processed_batch = []
            for list_element in output_batch:
                processed_batch.append(self.post_process_output(list_element))
        elif isinstance(output_batch, dict):
            processed_batch = {}
            for key in output_batch.keys():
                processed_batch[key] = self.post_process_output(output_batch[key])
        else:
            raise RuntimeError(f'How do I post process an object of type: {type(output_batch)}?')
        return processed_batch

    @staticmethod
    def post_process_pred_frame(pred_frame):
        uint8_frame = numpy.round(pred_frame * 255).astype('uint8')
        return uint8_frame

    @staticmethod
    def post_process_mask(mask: numpy.ndarray):
        bool_mask = mask[:, :, 0].astype('bool')
        return bool_mask

    @staticmethod
    def read_image(path: Path):
        image = skimage.io.imread(path.as_posix())
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
        mask = skimage.io.imread(path.as_posix()) == 255
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
            raise RuntimeError(f'Unknown mask format: {path.as_posix()}')
        return

    @staticmethod
    def read_depth(path: Path) -> numpy.ndarray:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['arr_0']
        elif path.suffix == '.exr':
            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.suffix}')
        return depth

    @staticmethod
    def save_flow(path: Path, flow: numpy.ndarray, as_png: bool = False):
        flow_image = flow_vis.flow_to_color(flow[:, :, :2])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), flow_image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), flow)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), flow_image)
        elif path.suffix == '.npz':
            numpy.savez_compressed(path.as_posix(), flow)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), flow_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown flow format: {path.as_posix()}')
        return


def save_configs(output_dirpath: Path, configs: dict, filename='Configs'):
    configs_path = output_dirpath / f'{filename}.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            from deepdiff import DeepDiff
            print(DeepDiff(old_configs, configs))
            raise RuntimeError(f'Configs mismatch while resuming testing: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_testing(train_configs: dict, test_configs: dict, save_intermediate_results: bool = False):
    test_num = test_configs['test_num']
    root_dirpath = Path('../')
    pred_frames_dirname = 'PredictedFrames'
    pred_masks_dirname = 'PredictedFramesMask'

    output_dirpath = root_dirpath / f'Runs/Testing/Test{test_num:04}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, train_configs, filename='TrainConfigs')
    save_configs(output_dirpath, test_configs, filename='TestConfigs')
    train_configs = train_configs.copy()
    train_configs['frame_predictor']['num_mpi_planes'] = test_configs['num_mpi_planes']
    test_configs = test_configs.copy()
    train_configs['device'] = test_configs['device']
    test_configs['root_dirpath'] = root_dirpath

    database_dirpath = root_dirpath / f'Data/Databases/{test_configs["database_name"]}'

    test_set_num = test_configs['test_set_num']
    frames_datapath = root_dirpath / f'res/TrainTestSets/{test_configs["database_name"]}/Set{test_set_num:02}/TestVideosData.csv'
    frames_data = pandas.read_csv(frames_datapath)

    tester = Tester(root_dirpath, database_dirpath, train_configs, device=test_configs['device'])
    flow_estimation_train_num = test_configs['flow_estimation']['train_num']
    flow_estimation_model_name = test_configs['flow_estimation']['model_name']
    video_inpainting_train_num = test_configs['video_inpainting']['train_num']
    video_inpainting_model_name = test_configs['video_inpainting']['model_name']
    flow_estimation_model_path = root_dirpath / f'Runs/Training/Train{flow_estimation_train_num:04}/SavedModels/Model_{flow_estimation_model_name}.tar'
    video_inpainting_model_path = root_dirpath / f'Runs/Training/Train{video_inpainting_train_num:04}/SavedModels/Model_{video_inpainting_model_name}.tar'
    tester.frame_predictor.load_weights(flow_estimation_model_path, video_inpainting_model_path)
    tester.pre_test_ops()

    print(f'Testing begins for Test{test_num:04}')
    num_frames = frames_data.shape[0]
    for i, frame_data in tqdm(frames_data.iterrows(), total=num_frames):
        video_name, seq_num, pred_frame_num = frame_data

        if isinstance(pred_frame_num, str):
            pred_frame_num = literal_eval(pred_frame_num)[0]
        if isinstance(seq_num, int):
            seq_name = f'seq{seq_num:02}'
        elif isinstance(seq_num, str):
            seq_name = seq_num
        else:
            raise RuntimeError

        video_output_dirpath = output_dirpath / f'{video_name}/{seq_name}'
        pred_frames_dirpath = video_output_dirpath / pred_frames_dirname
        pred_frames_mask_dirpath = video_output_dirpath / pred_masks_dirname
        pred_frames_dirpath.mkdir(parents=True, exist_ok=True)
        pred_frames_mask_dirpath.mkdir(parents=True, exist_ok=True)
        frame1_op_path = pred_frames_dirpath / f'{pred_frame_num - 1:04}.png'
        pred_frame2_path = pred_frames_dirpath / f'{pred_frame_num:04}.png'

        if not pred_frame2_path.exists():
            pred_frames, masks, extras = tester.predict_next_frame2(video_name, seq_num, pred_frame_num,
                                                                    save_intermediate_results)
            for i in range(len(pred_frames)):
                pred_frame_path = pred_frames_dirpath / f'{pred_frame_num + i:04}.png'
                pred_frame_mask_path = pred_frames_mask_dirpath / f'{pred_frame_num + i:04}.png'
                tester.save_image(pred_frame_path, pred_frames[i])
                tester.save_mask(pred_frame_mask_path, masks[i])

                if save_intermediate_results:
                    local_flow12_path = video_output_dirpath / f'LocalPredictedFlows/{pred_frame_num - 1 + i:04}.npy'
                    local_warped_frame2_path = video_output_dirpath / f'LocalFlowWarpedFrames/{pred_frame_num + i:04}.png'
                    total_warped_frame2_path = video_output_dirpath / f'TotalFlowWarpedFrames/{pred_frame_num + i:04}.png'
                    tester.save_flow(local_flow12_path, extras['local_flow12'], as_png=True)
                    tester.save_image(local_warped_frame2_path, extras['local_flow_warped_frame2'])
                    tester.save_image(total_warped_frame2_path, extras['total_flow_warped_frame2'])
    torch.cuda.empty_cache()
    return output_dirpath
