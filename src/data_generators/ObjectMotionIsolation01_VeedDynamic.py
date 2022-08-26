# Shree KRISHNAya Namaha
# Saves warped frame, mask, warped_depth.
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import time
import traceback
from pathlib import Path
from typing import List

import numpy
import pandas
from tqdm import tqdm

from utils.Warper import Warper

this_filepath = Path(__file__)
this_filename = this_filepath.stem


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


def start_generation(frames_data: pandas.DataFrame, num_steps: int):
    root_dirpath = Path('../../')
    data_dirpath = root_dirpath / 'Data/Databases/VeedDynamic/all_short'

    step_folder_name = f'{abs(num_steps)}step_'
    if num_steps > 0:
        step_folder_name += 'forward'
    else:
        step_folder_name += 'backward'

    warper = Warper(resolution=(1080, 1920))
    rendered_dirpath = data_dirpath / 'RenderedData'
    output_dirpath = data_dirpath / 'PoseWarping/PoseWarping01'

    for i, frame_data in tqdm(frames_data.iterrows(), total=frames_data.shape[0]):
        video_name, seq_num, frame1_num = frame_data

        warped_frames_dirpath = output_dirpath / f'{video_name}/seq{seq_num:02}/{step_folder_name}/warped_frames'
        warped_depths_dirpath = output_dirpath / f'{video_name}/seq{seq_num:02}/{step_folder_name}/warped_depths'
        mask_frames_dirpath = output_dirpath / f'{video_name}/seq{seq_num:02}/{step_folder_name}/masks'
        warped_frames_dirpath.mkdir(parents=True, exist_ok=True)
        warped_depths_dirpath.mkdir(parents=True, exist_ok=True)
        mask_frames_dirpath.mkdir(parents=True, exist_ok=True)

        transformation_path = rendered_dirpath / f'{video_name}/seq{seq_num:02}/TransformationMatrices.csv'
        transformation_matrices = numpy.genfromtxt(transformation_path, delimiter=',')

        frame2_num = frame1_num + num_steps
        frame1_path = rendered_dirpath / f'{video_name}/seq{seq_num:02}/rgb/{frame1_num:04}.png'
        depth1_path = rendered_dirpath / f'{video_name}/seq{seq_num:02}/depth/{frame1_num:04}.exr'

        warp_frame_path = warped_frames_dirpath / f'{frame2_num:04}.npy'
        warp_depth_path = warped_depths_dirpath / f'{frame2_num:04}.npy'
        mask_frame_path = mask_frames_dirpath / f'{frame2_num:04}.npy'

        if warp_frame_path.exists() and mask_frame_path.exists() and warp_depth_path.exists():
            continue

        frame1 = warper.read_image(frame1_path)
        depth1 = warper.read_depth(depth1_path)
        intrinsic = camera_intrinsic_transform()
        transformation1 = transformation_matrices[frame1_num].reshape(4, 4)
        transformation2 = transformation_matrices[frame2_num].reshape(4, 4)

        warped_frame2, mask2, warped_depth2, _ = warper.forward_warp(frame1, None, depth1, transformation1, transformation2, intrinsic, intrinsic)

        if not warp_frame_path.exists():
            warper.save_image(warp_frame_path, warped_frame2, as_png=True)

        if not mask_frame_path.exists():
            warper.save_mask(mask_frame_path, mask2, as_png=True)

        if not warp_depth_path.exists():
            warper.save_depth(warp_depth_path, warped_depth2)
    return


def create_frames_data01(video_names: List[str], seq_nos: List[int], frame1_nos: List[int]) -> pandas.DataFrame:
    frames_data = []
    for video_name in video_names:
        for seq_num in seq_nos:
            for frame1 in frame1_nos:
                frames_data.append([video_name, seq_num, frame1])
    frames_data = pandas.DataFrame(frames_data, columns=['video_name', 'seq_num', 'frame1_num'])
    return frames_data


def wrapper01(group: str, set_num: int, frame1_start: int, frame1_end: int, frame1_step: int, num_steps: int):
    """
    A wrapper that generates frames_data
    :param group: One of ['train', 'validation', 'test']
    :param set_num:
    :param frame1_start:
    :param frame1_end:
    :param frame1_step:
    :param num_steps:
    :return:
    """
    frame1_nos = list(range(frame1_start, frame1_end + 1, frame1_step))
    seq_nos = list(range(4))

    group_videos_datapath = Path(f'../../res/TrainTestSets/VeedDynamic/Set{set_num:02}/{group.capitalize()}VideosData.csv')
    group_video_names = numpy.unique(pandas.read_csv(group_videos_datapath)['video_name'])

    frames_data = create_frames_data01(group_video_names, seq_nos, frame1_nos)
    start_generation(frames_data, num_steps)
    return


def main():
    wrapper01(group='train', set_num=1, frame1_start=0, frame1_end=9, frame1_step=1, num_steps=2)
    wrapper01(group='validation', set_num=1, frame1_start=0, frame1_end=9, frame1_step=1, num_steps=2)
    wrapper01(group='test', set_num=1, frame1_start=7, frame1_end=9, frame1_step=2, num_steps=2)

    wrapper01(group='train', set_num=1, frame1_start=0, frame1_end=9, frame1_step=1, num_steps=-1)
    wrapper01(group='validation', set_num=1, frame1_start=0, frame1_end=9, frame1_step=1, num_steps=-1)
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
