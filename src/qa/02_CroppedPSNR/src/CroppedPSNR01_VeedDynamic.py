# Shree KRISHNAya Namaha
# PSNR measure between cropped predicted frames and ground truth frames
# Modified from QA07/CroppedRMSE01_OursBlender.py
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import argparse
import datetime
import json
import time
import traceback
from ast import literal_eval
from pathlib import Path

import numpy
import pandas
import simplejson
import skimage.io
import skvideo.io
from tqdm import tqdm

this_filepath = Path(__file__)
this_filename = Path(__file__).stem
this_metric_name = this_filename[:-12]


class PSNR:
    def __init__(self, frames_data: pandas.DataFrame, verbose_log: bool = True) -> None:
        super().__init__()
        self.frames_data = self.process_frames_data(frames_data)
        self.verbose_log = verbose_log
        return

    @staticmethod
    def process_frames_data(frames_data: pandas.DataFrame):
        if isinstance(frames_data['pred_frame_num'].to_numpy()[0], int):
            return frames_data
        if isinstance(frames_data['pred_frame_num'].to_numpy()[0].item(), int):
            return frames_data
        # Each pred_frame_num is a list of ints. Unpack them
        processed_frames_data = []
        for i, frame_data in frames_data.iterrows():
            video_name, seq_num, pred_frame_nums = frame_data
            pred_frame_nums = literal_eval(pred_frame_nums)
            for pred_frame_num in pred_frame_nums:
                processed_frames_data.append([video_name, seq_num, pred_frame_num])
        processed_frames_data = pandas.DataFrame(processed_frames_data, columns=frames_data.columns)
        return processed_frames_data

    @staticmethod
    def compute_frame_psnr(gt_frame: numpy.ndarray, eval_frame: numpy.ndarray):
        gt_frame = gt_frame[40:-40, 60:-60]
        eval_frame = eval_frame[40:-40, 60:-60]
        error = gt_frame.astype('float') - eval_frame.astype('float')
        mse = numpy.mean(numpy.square(error))
        psnr = 10 * numpy.log10(255**2 / mse)
        return psnr

    def compute_avg_psnr(self, old_data: pandas.DataFrame, database_dirpath: Path, pred_videos_dirpath: Path,
                         pred_folder_name: str):
        """

        :param old_data:
        :param database_dirpath: Should be path to Databases/OursBlender/Data
        :param pred_videos_dirpath:
        :param pred_folder_name:
        :return:
        """
        qa_scores = []
        for i, frame_data in tqdm(self.frames_data.iterrows(), total=self.frames_data.shape[0], leave=self.verbose_log):
            video_name, seq_num, pred_frame_num = frame_data
            if old_data is not None and old_data.loc[
                (old_data['video_name'] == video_name) & (old_data['seq_num'] == seq_num) &
                (old_data['pred_frame_num'] == pred_frame_num)
            ].size > 0:
                continue
            gt_frame_path = database_dirpath / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/rgb/{pred_frame_num:04}.png'
            gt_frame = skimage.io.imread(gt_frame_path.as_posix())
            pred_frame_path = pred_videos_dirpath / f'{video_name}/seq{seq_num:02}/{pred_folder_name}/{pred_frame_num:04}.png'
            pred_frame = skimage.io.imread(pred_frame_path.as_posix())
            qa_score = self.compute_frame_psnr(gt_frame, pred_frame)
            qa_scores.append([video_name, seq_num, pred_frame_num, qa_score])
        qa_scores_data = pandas.DataFrame(qa_scores, columns=['video_name', 'seq_num', 'pred_frame_num', this_metric_name])

        merged_data = self.update_qa_frame_data(old_data, qa_scores_data)
        merged_data = merged_data.round({this_metric_name: 4, })

        avg_psnr = numpy.mean(merged_data[this_metric_name])
        if isinstance(avg_psnr, numpy.ndarray):
            avg_psnr = avg_psnr.item()
        avg_psnr = numpy.round(avg_psnr, 4)
        return avg_psnr, merged_data

    @staticmethod
    def update_qa_frame_data(old_data: pandas.DataFrame, new_data: pandas.DataFrame):
        if (old_data is not None) and (new_data.size > 0):
            old_data = old_data.copy()
            new_data = new_data.copy()
            old_data.set_index(['video_name', 'seq_num', 'pred_frame_num'], inplace=True)
            new_data.set_index(['video_name', 'seq_num', 'pred_frame_num'], inplace=True)
            merged_data = old_data.combine_first(new_data).reset_index()
        elif old_data is not None:
            merged_data = old_data
        else:
            merged_data = new_data
        return merged_data


# noinspection PyUnusedLocal
def start_qa(pred_videos_dirpath: Path, database_dirpath: Path, frames_datapath: Path, pred_folder_name: str):
    if not pred_videos_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: pred_videos_dirpath does not exist')
        return

    if not database_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: database_dirpath does not exist')
        return

    qa_scores_filepath = pred_videos_dirpath / 'QA_Scores.json'
    psnr_data_path = pred_videos_dirpath / f'QA_Scores/{pred_folder_name}/{this_metric_name}_FrameWise.csv'
    if qa_scores_filepath.exists():
        with open(qa_scores_filepath.as_posix(), 'r') as qa_scores_file:
            qa_scores = json.load(qa_scores_file)
    else:
        qa_scores = {}

    if pred_folder_name in qa_scores:
        if this_metric_name in qa_scores[pred_folder_name]:
            avg_psnr = qa_scores[pred_folder_name][this_metric_name]
            print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_folder_name}: {avg_psnr}')
            return
    else:
        qa_scores[pred_folder_name] = {}

    if psnr_data_path.exists():
        psnr_data = pandas.read_csv(psnr_data_path)
    else:
        psnr_data = None

    frames_data = pandas.read_csv(frames_datapath)

    mse_computer = PSNR(frames_data)
    avg_psnr, psnr_data = mse_computer.compute_avg_psnr(psnr_data, database_dirpath, pred_videos_dirpath, pred_folder_name)
    qa_scores[pred_folder_name][this_metric_name] = avg_psnr
    print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {pred_folder_name}: {avg_psnr}')
    with open(qa_scores_filepath.as_posix(), 'w') as qa_scores_file:
        simplejson.dump(qa_scores, qa_scores_file, indent=4)
    psnr_data_path.parent.mkdir(parents=True, exist_ok=True)
    psnr_data.to_csv(psnr_data_path, index=False)
    return


def demo1():
    pred_videos_dirpath = Path('../../../VideoPrediction/Literature/003_MCnet/Runs/Testing/Test0006')
    database_dirpath = Path('../../../../Databases/OursBlender/Data')
    frames_data_path = Path('../../../../Databases/OursBlender/Data/TrainTestSets/Set01/TestVideosData.csv')
    pred_folder_name = 'PredictedFrames'
    start_qa(pred_videos_dirpath, database_dirpath, frames_data_path, pred_folder_name)
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

    start_qa(pred_videos_dirpath, database_dirpath, frames_datapath, pred_folder_name)
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
        'pred_folder_name':args.pred_folder_name,
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

    if args['email_id'] is not None:
        from snb_utils import Mailer

        receiver_address = args['email_id']
        subject = f'QA/{this_filename}'
        mail_content = f'{this_filename} has finished.\n' + run_result
        Mailer.send_mail(subject, mail_content, receiver_address)
