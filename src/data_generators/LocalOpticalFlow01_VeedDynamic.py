# Shree KRISHNAya Namaha
# Estimates and saves local optical flow from a rendered frame n to warped frame n-k.
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import inspect
import time
import traceback
from pathlib import Path
from typing import List

import numpy
import pandas
import simplejson
import torch
from deepdiff import DeepDiff
from tqdm import tqdm

import flow_estimation.Tester01 as Tester
from utils import CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem
this_filenum = int(this_filename[16:18])


class FlowEstimator(Tester.Tester):
    def __init__(self, root_dirpath: Path, database_dirpath: Path, train_configs: dict, pw_step_folder_name: str, device: str):
        super().__init__(root_dirpath, database_dirpath, train_configs, device=device)
        self.pw_step_folder_name = pw_step_folder_name
        return

    def load_data(self, video_name, seq_num, frame1_num):
        input_data = self.data_loader.load_generation_data(video_name, seq_num, frame1_num, self.pw_step_folder_name)
        return input_data

    def estimate_flow1(self, input_batch: dict):
        CommonUtils.move_to_device(input_batch, self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_batch = self.frame_predictor(input_batch)
        processed_input = self.post_process_output(input_batch)
        processed_output = self.post_process_output(output_batch)

        est_flow12 = self.post_process_est_flow2(processed_output['estimated_mpi_flows12'][0][0],
                                                 processed_input['mpi1_alpha'],
                                                 processed_input['mpi1_depth_planes'])
        return est_flow12

    def estimate_flow2(self, video_name, seq_num, frame1_num):
        input_dict = self.load_data(video_name, seq_num, frame1_num)
        est_flow12 = self.estimate_flow1(input_dict)
        return est_flow12

    @staticmethod
    def post_process_est_flow2(mpi_flow, mpi_alpha, mpi_depth):
        """

        :param mpi_flow: (h, w, d, 2)
        :param mpi_alpha: (h, w, d, 1)
        :param mpi_depth: (h, w, d, 1)
        :return:
        """
        composited_flow = numpy.sum(mpi_flow * mpi_alpha, axis=2)  # (h, w, 2+d)
        composited_depth = numpy.sum(mpi_depth * mpi_alpha, axis=2)  # (h, w, 1)
        flow12xy = composited_flow[:, :, :2]
        trans_depth = numpy.sum(composited_flow[:, :, 2:] * mpi_depth[:, :, :, 0], axis=2)[:, :, None]  # (h, w, 1)
        flow12z = trans_depth - composited_depth
        flow12 = numpy.concatenate([flow12xy, flow12z], axis=2)
        return flow12


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
    num_steps = gen_configs['num_steps']
    train_num = gen_configs['train_num']

    root_dirpath = Path('../../')
    data_dirpath = root_dirpath / 'Data/Databases/VeedDynamic'
    output_dirpath = data_dirpath / f'all_short/LocalOpticalFlows/LOF_{gen_num:02}_{gen_configs["description"]}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, gen_configs)
    gen_configs = gen_configs.copy()
    gen_configs['root_dirpath'] = root_dirpath

    train_dirpath = root_dirpath / f'Runs/Training/Train{train_num:04}'
    train_configs_path = train_dirpath / 'Configs.json'
    with open(train_configs_path.as_posix(), 'r') as configs_file:
        train_configs = simplejson.load(configs_file)
    train_configs['root_dirpath'] = root_dirpath
    train_configs['flow_estimation']['data_loader']['num_mpi_planes'] = gen_configs['num_mpi_planes']
    train_configs['device'] = gen_configs['device']

    if num_steps > 0:
        pw_step_folder_name = f'{num_steps}step_backward'
    else:
        pw_step_folder_name = f'{-num_steps}step_forward'

    estimator = FlowEstimator(root_dirpath, data_dirpath, train_configs, pw_step_folder_name, gen_configs['device'])
    estimator.load_model(gen_configs['model_name'])
    estimator.pre_test_ops()

    for i, frame_data in tqdm(frames_data.iterrows(), total=frames_data.shape[0]):
        video_name, seq_num, frame1_num = frame_data

        flow_dirpath = output_dirpath / f'{video_name}/seq{seq_num:02}/flows'
        # flow_masks_dirpath = output_dirpath / f'{video_name}/seq{seq_num:02}/flow_masks'
        flow_dirpath.mkdir(parents=True, exist_ok=True)
        # flow_masks_dirpath.mkdir(parents=True, exist_ok=True)

        flow12_path = flow_dirpath / f'{frame1_num:04}.npy'
        # flow12_mask_path = flow_masks_dirpath / f'{frame1_num:04}.npy'

        if flow12_path.exists():
            continue

        flow12 = estimator.estimate_flow2(video_name, seq_num, frame1_num)
        estimator.save_flow(flow12_path, flow12)
        # estimator.save_mask(flow12_mask_path, flow12_mask, as_png=True)
    return


def create_frames_data01(video_names: List[str], seq_nos: List[int], frame1_nos: List[int]) -> pandas.DataFrame:
    frames_data = []
    for video_name in video_names:
        for seq_num in seq_nos:
            for frame1 in frame1_nos:
                frames_data.append([video_name, seq_num, frame1])
    frames_data = pandas.DataFrame(frames_data, columns=['video_name', 'seq_num', 'frame1_num'])
    return frames_data


def wrapper01(gen_num: int, description: str, train_num: int, set_num: int, num_steps: int,
              frame1_start: int, frame1_end: int, frame1_step: int, group: str):
    """
    A wrapper that generates frames_data
    :param gen_num:
    :param description:
    :param train_num: 
    :param set_num:
    :param num_steps:
    :param frame1_start: 
    :param frame1_end: 
    :param frame1_step: 
    :param group: One of ['train', 'validation', 'test']
    :return:
    """
    this_method = inspect.currentframe().f_code.co_name
    configs = {
        'DataGenerator': f'{this_filename}/{Tester.this_filename}/{this_method}',
        'gen_num': gen_num,
        'description': description,
        'gen_set_num': set_num,
        'num_steps': num_steps,
        'train_num': train_num,
        'model_name': 'Iter010000',
        'database_name': 'VeedDynamic',
        'num_mpi_planes': 4,
        'device': 'gpu0',
    }

    frame1_nos = list(range(frame1_start, frame1_end + 1, frame1_step))
    seq_nos = list(range(4))
    group_videos_datapath = Path(f'../../res/TrainTestSets/VeedDynamic/Set{set_num:02}/{group.capitalize()}VideosData.csv')
    group_video_names = numpy.unique(pandas.read_csv(group_videos_datapath)['video_name'])

    frames_data = create_frames_data01(group_video_names, seq_nos, frame1_nos)
    start_generation(configs, frames_data)
    return


def main():
    wrapper01(gen_num=1, description='2step_backward', train_num=1, set_num=1, num_steps=-2, frame1_start=2, frame1_end=10, frame1_step=1, group='train')
    wrapper01(gen_num=1, description='2step_backward', train_num=1, set_num=1, num_steps=-2, frame1_start=2, frame1_end=10, frame1_step=1, group='validation')
    wrapper01(gen_num=1, description='2step_backward', train_num=1, set_num=1, num_steps=-2, frame1_start=6, frame1_end=8, frame1_step=1, group='test')
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
