# Shree KRISHNAya Namaha
# Trains and tests and runs QA
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import os
import sys
import time
import traceback
from pathlib import Path

import Tester01 as Tester

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def start_testing(train_configs: dict, test_configs: dict):
    output_dirpath = Tester.start_testing(train_configs, test_configs, save_intermediate_results=False)

    root_dirpath = Path('../')
    database_dirpath = root_dirpath / f'Data/Databases/{test_configs["database_name"]}'
    test_set_num = test_configs['test_set_num']
    frames_datapath = root_dirpath / f'res/TrainTestSets/{test_configs["database_name"]}/Set{test_set_num:02}/TestVideosData.csv'

    pred_folder_name = 'PredictedFrames'

    # Run QA
    qa_filepath = root_dirpath / Path('QA/00_Common/src/AllMetrics02_Sintel.py')
    cmd = f'python {qa_filepath.absolute().as_posix()} ' \
          f'--demo_function_name demo2 ' \
          f'--pred_videos_dirpath {output_dirpath.absolute().as_posix()} ' \
          f'--database_dirpath {database_dirpath.absolute().as_posix()} ' \
          f'--frames_datapath {frames_datapath.absolute().as_posix()} ' \
          f'--pred_folder_name {pred_folder_name}'
    os.system(cmd)
    return


def demo1():
    test_num = 101
    flow_estimation_train_num = 101
    video_inpainting_train_num = 102

    train_configs = {
        'database': 'MPI_Sintel',
        'data_loader': {
            'name': 'Sintel01',
            'train_set_num': 1,
            'load_true_flow': False,
            'load_target_frames': False,
            'num_mpi_planes': 4,
        },
        'flow_estimation': {
            'feature_extractor': {
                'name': 'PyramidFeatureExtractor01',
            },
            'flow_estimator': {
                'name': 'PWCnet01',
                'upsample_flow': True,
                'search_range_spatial': 4,
                'search_range_depth': 1,
            },
        },
        'video_inpainting': {
            'flow_predictor': {
                'name': 'Unet01',
            },
        },
        'local_flow_predictor': {
            'name': 'LinearPredictor01',
        },
        'frame_predictor': {
            'name': 'FramePredictor01',
            'num_pred_frames': 1,
            'num_infilling_iterations': 3,
        },
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 1,
        'flow_estimation': {
            'train_num': flow_estimation_train_num,
            'model_name': 'Iter010000',
        },
        'video_inpainting': {
            'train_num': video_inpainting_train_num,
            'model_name': 'Iter010000',
        },
        'database_name': 'MPI_Sintel',
        'num_mpi_planes': 4,
        'device': 'gpu0',
    }
    start_testing(train_configs, test_configs)
    return


def demo2():
    test_num = 102
    flow_estimation_train_num = 101
    video_inpainting_train_num = 102

    train_configs = {
        'database': 'MPI_Sintel',
        'data_loader': {
            'name': 'Sintel01',
            'train_set_num': 1,
            'load_true_flow': True,
            'load_target_frames': False,
            'num_mpi_planes': 4,
        },
        'flow_estimation': {
            'feature_extractor': {
                'name': 'PyramidFeatureExtractor01',
            },
            'flow_estimator': {
                'name': 'PWCnet01',
                'upsample_flow': True,
                'search_range_spatial': 4,
                'search_range_depth': 1,
            },
        },
        'video_inpainting': {
            'flow_predictor': {
                'name': 'Unet01',
            },
        },
        'local_flow_predictor': {
            'name': 'LinearPredictor01',
        },
        'frame_predictor': {
            'name': 'FramePredictor02',
            'num_pred_frames': 1,
            'num_infilling_iterations': 3,
        },
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 1,
        'flow_estimation': {
            'train_num': flow_estimation_train_num,
            'model_name': 'Iter010000',
        },
        'video_inpainting': {
            'train_num': video_inpainting_train_num,
            'model_name': 'Iter010000',
        },
        'database_name': 'MPI_Sintel',
        'num_mpi_planes': 4,
        'device': 'gpu0',
    }
    start_testing(train_configs, test_configs)
    return


def demo3():
    test_num = 103
    flow_estimation_train_num = 101
    video_inpainting_train_num = 102

    train_configs = {
        'database': 'MPI_Sintel',
        'data_loader': {
            'name': 'Sintel01',
            'train_set_num': 1,
            'load_true_flow': False,
            'load_target_frames': True,
            'num_mpi_planes': 4,
        },
        'flow_estimation': {
            'feature_extractor': {
                'name': 'PyramidFeatureExtractor01',
            },
            'flow_estimator': {
                'name': 'PWCnet01',
                'upsample_flow': True,
                'search_range_spatial': 4,
                'search_range_depth': 1,
            },
        },
        'video_inpainting': {
            'flow_predictor': {
                'name': 'Unet01',
            },
        },
        'local_flow_predictor': {
            'name': 'LinearPredictor01',
        },
        'frame_predictor': {
            'name': 'FramePredictor03',
            'num_pred_frames': 1,
            'num_infilling_iterations': 3,
        },
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 1,
        'flow_estimation': {
            'train_num': flow_estimation_train_num,
            'model_name': 'Iter010000',
        },
        'video_inpainting': {
            'train_num': video_inpainting_train_num,
            'model_name': 'Iter010000',
        },
        'database_name': 'MPI_Sintel',
        'num_mpi_planes': 4,
        'device': 'gpu0',
    }
    start_testing(train_configs, test_configs)
    return


def demo4():
    test_num = 104
    flow_estimation_train_num = 101
    video_inpainting_train_num = 102

    train_configs = {
        'database': 'MPI_Sintel',
        'data_loader': {
            'name': 'Sintel01',
            'train_set_num': 1,
            'load_true_flow': True,
            'load_target_frames': True,
            'num_mpi_planes': 4,
        },
        'flow_estimation': {
            'feature_extractor': {
                'name': 'PyramidFeatureExtractor01',
            },
            'flow_estimator': {
                'name': 'PWCnet01',
                'upsample_flow': True,
                'search_range_spatial': 4,
                'search_range_depth': 1,
            },
        },
        'video_inpainting': {
            'flow_predictor': {
                'name': 'Unet01',
            },
        },
        'local_flow_predictor': {
            'name': 'LinearPredictor01',
        },
        'frame_predictor': {
            'name': 'FramePredictor04',
            'num_pred_frames': 1,
            'num_infilling_iterations': 3,
        },
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 1,
        'flow_estimation': {
            'train_num': flow_estimation_train_num,
            'model_name': 'Iter010000',
        },
        'video_inpainting': {
            'train_num': video_inpainting_train_num,
            'model_name': 'Iter010000',
        },
        'database_name': 'MPI_Sintel',
        'num_mpi_planes': 4,
        'device': 'gpu0',
    }
    start_testing(train_configs, test_configs)
    return


def demo5():
    test_num = 105
    flow_estimation_train_num = 101
    video_inpainting_train_num = 102

    train_configs = {
        'database': 'MPI_Sintel',
        'data_loader': {
            'name': 'Sintel01',
            'train_set_num': 2,
            'load_true_flow': False,
            'load_target_frames': False,
            'num_mpi_planes': 4,
        },
        'flow_estimation': {
            'feature_extractor': {
                'name': 'PyramidFeatureExtractor01',
            },
            'flow_estimator': {
                'name': 'PWCnet01',
                'upsample_flow': True,
                'search_range_spatial': 4,
                'search_range_depth': 1,
            },
        },
        'video_inpainting': {
            'flow_predictor': {
                'name': 'Unet01',
            },
        },
        'local_flow_predictor': {
            'name': 'LinearPredictor01',
        },
        'frame_predictor': {
            'name': 'FramePredictor01',
            'num_pred_frames': 4,
            'num_infilling_iterations': 3,
        },
    }
    test_configs = {
        'Tester': f'{this_filename}/{Tester.this_filename}',
        'test_num': test_num,
        'test_set_num': 2,
        'flow_estimation': {
            'train_num': flow_estimation_train_num,
            'model_name': 'Iter010000',
        },
        'video_inpainting': {
            'train_num': video_inpainting_train_num,
            'model_name': 'Iter010000',
        },
        'database_name': 'MPI_Sintel',
        'num_mpi_planes': 4,
        'device': 'gpu0',
    }
    start_testing(train_configs, test_configs)
    return


def main():
    demo1()
    demo2()
    demo3()
    demo4()
    demo5()
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
