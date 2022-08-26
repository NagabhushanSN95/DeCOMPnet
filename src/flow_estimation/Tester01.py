# Shree KRISHNAya Namaha
# Tester for flow estimation model
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

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

from flow_estimation.data_loaders.DataLoaderFactory import get_data_loader
from flow_estimation.data_loaders.DataLoaderParent import DataLoaderParent
from flow_estimation.feature_extractors.FeatureExtractorFactory import get_feature_extractor
from flow_estimation.flow_estimators.FlowEstimatorFactory import get_flow_estimator
from flow_estimation.frame_predictors.FramePredictorFactory import get_frame_predictor
from utils import CommonUtils

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
        flow_estimator = None
        if 'feature_extractor' in self.train_configs['flow_estimation'].keys():
            feature_extractor = get_feature_extractor(self.train_configs)
        if 'flow_estimator' in self.train_configs['flow_estimation'].keys():
            flow_estimator = get_flow_estimator(self.train_configs)
        self.frame_predictor = get_frame_predictor(self.train_configs, feature_extractor, flow_estimator).to(self.device)
        return

    def get_data_loader(self):
        self.data_loader = get_data_loader(self.train_configs, self.database_dirpath, frames_datapath=None)
        return

    def load_model(self, model_name: str):
        train_num = self.train_configs["train_num"]
        full_model_name = f'Model_{model_name}.tar'
        train_dirpath = self.train_configs['root_dirpath'] / f'Runs/Training/Train{train_num:04}'
        saved_models_dirpath = train_dirpath / 'SavedModels'
        model_path = saved_models_dirpath / full_model_name
        checkpoint_state = torch.load(model_path, map_location=self.device)
        iter_num = checkpoint_state['iteration_num']
        self.frame_predictor.load_state_dict(checkpoint_state['model_state_dict'])
        print(f'Loaded Model in Train{train_num:04}/{full_model_name} trained for {iter_num} iterations')
        return

    def pre_test_ops(self):
        self.frame_predictor.eval()
        return

    def load_data(self, video_name, seq_num, pred_frame_num):
        input_data = self.data_loader.load_prediction_data(video_name, seq_num, pred_frame_num)
        return input_data

    def predict_next_frame1(self, input_batch: dict):
        CommonUtils.move_to_device(input_batch, self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_batch = self.frame_predictor(input_batch)
        processed_input = self.post_process_output(input_batch)
        processed_output = self.post_process_output(output_batch)

        pred_frame2 = self.post_process_pred_frame(processed_output['predicted_frame2'])
        mask2 = self.post_process_mask(processed_output['mask2'])
        est_flow12 = self.post_process_est_flow(processed_output['estimated_mpi_flows12'][0][0], processed_input['mpi1_alpha'])
        return pred_frame2, mask2, est_flow12

    def predict_next_frame2(self, video_name, seq_num, pred_frame_num):
        input_dict = self.load_data(video_name, seq_num, pred_frame_num)
        pred_frame2, mask2, est_flow12 = self.predict_next_frame1(input_dict)
        return pred_frame2, mask2, est_flow12

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
        uint8_frame = (pred_frame * 255).astype('uint8')
        return uint8_frame

    @staticmethod
    def post_process_est_flow(mpi_flow, mpi_alpha):
        flow_2d = numpy.sum(mpi_flow[:, :, :, :2] * mpi_alpha, axis=2)
        return flow_2d

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
        if path.suffix in ['.jpg', '.png', '.bmp']:
            skimage.io.imsave(path.as_posix(), image)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), image)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), image)
        else:
            raise RuntimeError(f'Unknown image format: {path.suffix}')
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
            raise RuntimeError(f'Unknown mask format: {path.suffix}')
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
        if path.suffix in ['.jpg', '.png', '.bmp']:
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
            raise RuntimeError(f'Unknown flow format: {path.suffix}')
        return


def save_configs(output_dirpath: Path, configs: dict):
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = simplejson.load(configs_file)
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs != old_configs:
            from deepdiff import DeepDiff
            print(DeepDiff(old_configs, configs))
            raise RuntimeError('Configs mismatch while resuming testing')
    else:
        with open(configs_path.as_posix(), 'w') as configs_file:
            simplejson.dump(configs, configs_file, indent=4)
    return


def start_testing(test_configs: dict):
    test_num = test_configs['test_num']
    root_dirpath = Path('../')
    pred_folder_name = 'ObjectMotionPredictedFrames'
    mask_folder_name = 'ObjectMotionMasks'
    flow_folder_name = 'ObjectMotionFlow'

    output_dirpath = root_dirpath / f'Runs/Testing/Test{test_num:04}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, test_configs)
    test_configs = test_configs.copy()
    test_configs['root_dirpath'] = root_dirpath

    train_dirpath = Path(f'../Runs/Training/Train{test_configs["train_num"]:04}')
    train_configs_path = train_dirpath / 'Configs.json'
    with open(train_configs_path.as_posix(), 'r') as configs_file:
        train_configs = simplejson.load(configs_file)
    train_configs['root_dirpath'] = root_dirpath
    train_configs['data_loader']['num_mpi_planes'] = test_configs['num_mpi_planes']
    train_configs['device'] = test_configs['device']

    database_dirpath = root_dirpath / f"../../../../Databases/{test_configs['database_dirpath']}"

    test_set_num = test_configs['test_set_num']
    frames_datapath = database_dirpath / f'TrainTestSets/Set{test_set_num:02}/TestVideosData.csv'
    frames_data = pandas.read_csv(frames_datapath)

    tester = Tester(root_dirpath, database_dirpath, train_configs, device=test_configs['device'])
    tester.load_model(test_configs['model_name'])
    tester.pre_test_ops()

    num_frames = frames_data.shape[0]
    for i, frame_data in tqdm(frames_data.iterrows(), total=num_frames):
        video_name, seq_num, pred_frame_num = frame_data

        video_output_dirpath = output_dirpath / f'{video_name}/seq{seq_num:02}'
        pred_frames_dirpath = video_output_dirpath / pred_folder_name
        masks_dirpath = video_output_dirpath / mask_folder_name
        flow_dirpath = video_output_dirpath / flow_folder_name
        pred_frames_dirpath.mkdir(parents=True, exist_ok=True)
        masks_dirpath.mkdir(parents=True, exist_ok=True)
        flow_dirpath.mkdir(parents=True, exist_ok=True)
        frame1_op_path = pred_frames_dirpath / f'{pred_frame_num - 1:04}.png'
        pred_frame2_path = pred_frames_dirpath / f'{pred_frame_num:04}.png'
        mask2_path = masks_dirpath / f'{pred_frame_num:04}.png'
        flow1_path = flow_dirpath / f'{pred_frame_num - 1:04}.npy'

        if 'resolution_suffix' in train_configs['data_loader']:
            resolution_suffix = train_configs['data_loader']['resolution_suffix']
        else:
            resolution_suffix = ''
        if not frame1_op_path.exists():
            frame1_path = database_dirpath / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/rgb{resolution_suffix}/{pred_frame_num - 1:04}.npy'
            # shutil.copy(frame1_path, frame1_op_path)
            frame1 = numpy.round(numpy.load(frame1_path.as_posix())).astype('uint8')
            skimage.io.imsave(frame1_op_path.as_posix(), frame1)

        if not pred_frame2_path.exists():
            pred_frame2, mask2, est_flow1 = tester.predict_next_frame2(video_name, seq_num, pred_frame_num)
            tester.save_image(pred_frame2_path, pred_frame2)
            tester.save_mask(mask2_path, mask2)
            tester.save_flow(flow1_path, est_flow1, as_png=True)
    torch.cuda.empty_cache()
    return output_dirpath
