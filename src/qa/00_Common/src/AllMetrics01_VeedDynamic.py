# Shree KRISHNAya Namaha
# Runs all metrics serially
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import argparse
import datetime
import importlib.util
import time
import traceback
from pathlib import Path

this_filepath = Path(__file__)
this_filename = Path(__file__).stem
this_metric_name = this_filename[:-7]


def run_all_qa(pred_videos_dirpath: Path, database_dirpath: Path, frames_datapath: Path, pred_folder_name: str):
    args_values = locals()
    metric_files = [
        this_filepath.parent / '../../01_CroppedRMSE/src/CroppedRMSE01_VeedDynamic.py',
        this_filepath.parent / '../../02_CroppedPSNR/src/CroppedPSNR01_VeedDynamic.py',
        this_filepath.parent / '../../03_CroppedSSIM/src/CroppedSSIM01_VeedDynamic.py',
        this_filepath.parent / '../../04_CroppedLPIPS/src/CroppedLPIPS01_VeedDynamic.py',
        this_filepath.parent / '../../05_CroppedSTRRED/src/CroppedSTRRED01_VeedDynamic.py',
    ]
    for metric_file_path in metric_files:
        spec = importlib.util.spec_from_file_location('module.name', metric_file_path.absolute().resolve().as_posix())
        qa_module = importlib.util.module_from_spec(spec)
        # noinspection PyUnresolvedReferences
        spec.loader.exec_module(qa_module)
        function_arguments = []
        for arg_name in run_all_qa.__code__.co_varnames:
            # noinspection PyUnresolvedReferences
            if arg_name in qa_module.start_qa.__code__.co_varnames:
                function_arguments.append(args_values[arg_name])
        # noinspection PyUnresolvedReferences
        qa_module.start_qa(*function_arguments)
    return


def demo1():
    root_dirpath = Path('../../../../')
    pred_videos_dirpath = root_dirpath / 'Runs/Testing/Test0001'
    database_dirpath = root_dirpath / 'Data/Databases/VeedDynamic/Data'
    frames_data_path = root_dirpath / 'res/TrainTestSets/VeedDynamic/Set01/TestVideosData.csv'
    pred_folder_name = 'PredictedFrames'
    run_all_qa(pred_videos_dirpath, database_dirpath, frames_data_path, pred_folder_name)
    return


def demo2(args: dict):
    pred_videos_dirpath = args['pred_videos_dirpath']
    if pred_videos_dirpath is None:
        raise RuntimeError(f'Please provide pred_videos_dirpath')
    pred_videos_dirpath = Path(pred_videos_dirpath)

    database_dirpath = args['database_dirpath']
    if database_dirpath is None:
        raise RuntimeError(f'Please provide database_dirpath')
    database_dirpath = Path(database_dirpath)

    frames_datapath = args['frames_datapath']
    if frames_datapath is None:
        raise RuntimeError(f'Please provide frames_datapath')
    frames_datapath = Path(frames_datapath)

    pred_folder_name = args['pred_folder_name']

    run_all_qa(pred_videos_dirpath, database_dirpath, frames_datapath, pred_folder_name)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_videos_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--frames_datapath')
    parser.add_argument('--pred_folder_name', default='PredictedFrames')
    parser.add_argument('--email_id')
    args = parser.parse_args()

    args_dict = {
        'demo_function_name': args.demo_function_name,
        'pred_videos_dirpath': args.pred_videos_dirpath,
        'database_dirpath': args.database_dirpath,
        'frames_datapath': args.frames_datapath,
        'pred_folder_name': args.pred_folder_name,
        'email_id': args.email_id,
    }
    return args_dict


def main(args: dict):
    if args['demo_function_name'] == 'demo1':
        demo1()
    elif args['demo_function_name'] == 'demo2':
        demo2(args)
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    args = parse_args()
    try:
        main(args)
        run_result = 'Program completed successfully!'
    except Exception as e:
        print(e)
        traceback.print_exc()
        run_result = "Error: " + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
