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


def start_generation(frames_data: pandas.DataFrame, num_steps: int):
    root_dirpath = Path('../../')
    data_dirpath = root_dirpath / 'Data/Databases/MPI_Sintel/all'

    step_folder_name = f'{abs(num_steps)}step_'
    if num_steps > 0:
        step_folder_name += 'forward'
    else:
        step_folder_name += 'backward'

    warper = Warper(resolution=(436, 1024))
    rendered_dirpath = data_dirpath / 'RenderedData'
    output_dirpath = data_dirpath / 'PoseWarping/PoseWarping01'

    for i, frame_data in tqdm(frames_data.iterrows(), total=frames_data.shape[0]):
        video_name, seq_name, frame1_num = frame_data

        warped_frames_dirpath = output_dirpath / f'{video_name}/{seq_name}/{step_folder_name}/warped_frames'
        warped_depths_dirpath = output_dirpath / f'{video_name}/{seq_name}/{step_folder_name}/warped_depths'
        mask_frames_dirpath = output_dirpath / f'{video_name}/{seq_name}/{step_folder_name}/masks'
        warped_frames_dirpath.mkdir(parents=True, exist_ok=True)
        warped_depths_dirpath.mkdir(parents=True, exist_ok=True)
        mask_frames_dirpath.mkdir(parents=True, exist_ok=True)

        transformation_path = rendered_dirpath / f'{video_name}/TransformationMatrices.csv'
        transformation_matrices = numpy.genfromtxt(transformation_path, delimiter=',')
        intrinsics_path = rendered_dirpath / f'{video_name}/CameraIntrinsics.csv'
        intrinsic_matrices = numpy.genfromtxt(intrinsics_path, delimiter=',')

        frame2_num = frame1_num + num_steps
        frame1_path = rendered_dirpath / f'{video_name}/rgb/{seq_name}/{frame1_num:04}.png'
        depth1_path = rendered_dirpath / f'{video_name}/depth/{frame1_num:04}.npz'

        warp_frame_path = warped_frames_dirpath / f'{frame2_num:04}.npy'
        warp_depth_path = warped_depths_dirpath / f'{frame2_num:04}.npy'
        mask_frame_path = mask_frames_dirpath / f'{frame2_num:04}.npy'

        if warp_frame_path.exists() and mask_frame_path.exists() and warp_depth_path.exists():
            continue

        frame1 = warper.read_image(frame1_path)
        depth1 = warper.read_depth(depth1_path)
        transformation1 = transformation_matrices[frame1_num].reshape(4, 4)
        transformation2 = transformation_matrices[frame2_num].reshape(4, 4)
        intrinsic1 = intrinsic_matrices[frame1_num].reshape(3, 3)
        intrinsic2 = intrinsic_matrices[frame2_num].reshape(3, 3)

        warped_frame2, mask2, warped_depth2, _ = warper.forward_warp(frame1, None, depth1, transformation1, transformation2, intrinsic1, intrinsic2)

        if not warp_frame_path.exists():
            warper.save_image(warp_frame_path, warped_frame2, as_png=True)

        if not mask_frame_path.exists():
            warper.save_mask(mask_frame_path, mask2, as_png=True)

        if not warp_depth_path.exists():
            warper.save_depth(warp_depth_path, warped_depth2)
    return


def create_frames_data01(video_names: List[str], seq_names: List[str], frame1_start: int, frame1_step: int,
                         num_frames_data: pandas.DataFrame) -> pandas.DataFrame:
    frames_data = []
    for video_name in video_names:
        for seq_name in seq_names:
            frame1_end = frame1_start + num_frames_data.loc[num_frames_data['video_name'] == video_name][
                'num_frames'].to_numpy()[0] - 4
            frame1_nos = list(range(frame1_start, frame1_end + 1, frame1_step))
            for frame1 in frame1_nos:
                frames_data.append([video_name, seq_name, frame1])
    frames_data = pandas.DataFrame(frames_data, columns=['video_name', 'seq_name', 'frame1_num'])
    return frames_data


def wrapper01(group: str, set_num: int, frame1_start: int, frame1_step: int, num_steps: int):
    """
    A wrapper that generates frames_data
    :param group: One of ['train', 'validation', 'test']
    :param set_num:
    :param frame1_start:
    :param frame1_step:
    :param num_steps:
    :return:
    """
    root_dirpath = Path('../../')
    database_dirpath = root_dirpath / 'Data/Databases/MPI_Sintel/all'
    num_frames_path = database_dirpath / 'NumberOfFramesPerVideo.csv'
    num_frames_data = pandas.read_csv(num_frames_path)

    group_videos_datapath = Path(f'../../res/TrainTestSets/MPI_Sintel/Set{set_num:02}/{group.capitalize()}VideosData.csv')
    group_video_names = numpy.unique(pandas.read_csv(group_videos_datapath)['video_name'])
    seq_names = numpy.unique(pandas.read_csv(group_videos_datapath)['seq_name'])

    frames_data = create_frames_data01(group_video_names, seq_names, frame1_start, frame1_step, num_frames_data)
    start_generation(frames_data, num_steps)
    return


def main():
    wrapper01(group='train', set_num=1, frame1_start=0, frame1_step=1, num_steps=2)
    wrapper01(group='validation', set_num=1, frame1_start=0, frame1_step=1, num_steps=2)
    wrapper01(group='test', set_num=1, frame1_start=7, frame1_step=2, num_steps=2)

    wrapper01(group='train', set_num=1, frame1_start=0, frame1_step=1, num_steps=-1)
    wrapper01(group='validation', set_num=1, frame1_start=0, frame1_step=1, num_steps=-1)
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
