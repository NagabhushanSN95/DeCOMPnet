# Shree KRISHNAya Namaha
# Generates mask for points of interest for local optical flow training.
# Given frames and their masks, a pixel is point of interest (POI), if either frame diff is greater than a threshold or
# one of the frames is unknown
# Modified from LOF_POI_01_VeedDynamic.py
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import inspect
import time
import traceback
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
from deepdiff import DeepDiff
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = this_filepath.stem


class PoiGenerator:
    def __init__(self, root_dirpath: Path, database_dirpath: Path, num_steps: int, intensity_threshold: int):
        self.root_dirpath = root_dirpath
        self.database_dirpath = database_dirpath
        self.pw_step_folder_name = self.get_pw_step_folder_name(num_steps)
        self.intensity_threshold = intensity_threshold
        return

    def load_data(self, video_name: str, seq_name: str, pred_frame_num: int):
        frame1_num = pred_frame_num - 1

        frame1_path = self.database_dirpath / f'RenderedData/{video_name}/rgb/{seq_name}/' \
                                              f'{frame1_num:04}.png'
        frame2_path = self.database_dirpath / f'PoseWarping/PoseWarping01/{video_name}/{seq_name}/' \
                                              f'{self.pw_step_folder_name}/warped_frames/{frame1_num:04}.npy'
        mask2_path = self.database_dirpath / f'PoseWarping/PoseWarping01/{video_name}/{seq_name}/' \
                                             f'{self.pw_step_folder_name}/masks/{frame1_num:04}.npy'
        
        frame1 = self.read_image(frame1_path)
        frame2 = self.read_image(frame2_path)
        mask2 = self.read_mask(mask2_path)
        mask1 = numpy.ones_like(mask2)
        
        data_dict = {
            'frame1': frame1,
            'mask1': mask1,
            'frame2': frame2,
            'mask2': mask2,
        }
        return data_dict
    
    @staticmethod
    def compute_poi_mask1(input_dict: dict):
        frame1 = input_dict['frame1']
        mask1 = input_dict['mask1']
        frame2 = input_dict['frame2']
        mask2 = input_dict['mask2']

        mask = mask1 & mask2
        frame_diff = frame1.astype('int') - frame2.astype('int')
        diff_mask = (numpy.abs(frame_diff) >= 10).any(axis=2)
        poi_mask = (~mask) | diff_mask
        return poi_mask

    def compute_poi_mask2(self, video_name: str, seq_name: str, pred_frame_num: int):
        data_dict = self.load_data(video_name, seq_name, pred_frame_num)
        poi_mask = self.compute_poi_mask1(data_dict)
        return poi_mask

    @staticmethod
    def get_pw_step_folder_name(num_steps: int):
        if num_steps > 0:
            pw_step_folder_name = f'{num_steps}step_backward'
        else:
            pw_step_folder_name = f'{-num_steps}step_forward'
        return pw_step_folder_name
    
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
    def read_mask(path: Path):
        if path.suffix == '.npy':
            mask = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown mask format: {path.as_posix()}')
        return mask

    @staticmethod
    def save_mask(path: Path, mask: numpy.ndarray, as_png: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        mask_image = mask.astype('uint8') * 255
        if path.suffix == '.png':
            skimage.io.imsave(path.as_posix(), mask_image, check_contrast=False)
        elif path.suffix == '.npy':
            numpy.save(path.as_posix(), mask)
            if as_png:
                png_path = path.parent / f'{path.stem}.png'
                skimage.io.imsave(png_path.as_posix(), mask_image, check_contrast=False)
        else:
            raise RuntimeError(f'Unknown mask format: {path.as_posix()}')
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
            raise RuntimeError(f'Configs mismatch while resuming generation: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_generation(gen_configs: dict, frames_data: pandas.DataFrame):
    gen_num = gen_configs['gen_num']
    num_steps = gen_configs['num_steps']
    intensity_threshold = gen_configs['intensity_threshold']

    root_dirpath = Path('../../')
    data_dirpath = root_dirpath / 'Data/Databases/MPI_Sintel/all'
    output_dirpath = data_dirpath / f"LOF_POI/LOF_POI_{gen_num:02}_{gen_configs['description']}"
    output_dirpath.mkdir(parents=True, exist_ok=True)
    save_configs(output_dirpath, gen_configs)
    gen_configs['root_dirpath'] = root_dirpath

    model = PoiGenerator(root_dirpath, data_dirpath, num_steps, intensity_threshold)

    for i, frame_data in tqdm(frames_data.iterrows(), total=frames_data.shape[0]):
        video_name, seq_name, pred_frame_num = frame_data
        frame1_num = pred_frame_num - 1

        output_path = output_dirpath / f'{video_name}/{seq_name}/{frame1_num:04}.npy'
        if output_path.exists():
            continue

        poi_mask = model.compute_poi_mask2(video_name, seq_name, pred_frame_num)
        model.save_mask(output_path, poi_mask, as_png=True)


def wrapper01(gen_num: int, description: str, num_steps: int, set_num: int, group: str):
    """
    A wrapper that generates frames_data
    :param gen_num:
    :param description:
    :param num_steps:
    :param set_num:
    :param group: One of ['train', 'validation', 'test']
    :return:
    """
    this_method = inspect.currentframe().f_code.co_name
    configs = {
        'DataGenerator': f'{this_filename}/{this_method}',
        'gen_num': gen_num,
        'description': description,
        'set_num': set_num,
        'num_steps': num_steps,
        'pose_warping_num': 1,
        'intensity_threshold': 10,
        'database_name': 'MPI_Sintel',
    }

    group_videos_datapath = Path(f'../../res/TrainTestSets/MPI_Sintel/Set{set_num:02}/{group.capitalize()}VideosData.csv')
    frames_data = pandas.read_csv(group_videos_datapath)
    start_generation(configs, frames_data)
    return


def main():
    wrapper01(gen_num=1, description='2step_backward', num_steps=-2, set_num=1, group='train')
    wrapper01(gen_num=1, description='2step_backward', num_steps=-2, set_num=1, group='validation')

    wrapper01(gen_num=2, description='1step_forward', num_steps=1, set_num=1, group='train')
    wrapper01(gen_num=2, description='1step_forward', num_steps=1, set_num=1, group='validation')
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
