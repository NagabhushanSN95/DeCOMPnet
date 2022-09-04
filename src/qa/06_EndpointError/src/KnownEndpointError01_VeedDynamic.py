# Shree KRISHNAya Namaha
# Computes average end-point error (AEPE) for local motion prediction
# Author: Nagabhushan S N
# Last Modified: 04/09/2022

import argparse
import datetime
import json
import time
import traceback
from pathlib import Path

import Imath
import OpenEXR
import numpy
import pandas
import simplejson
import skimage.io
import skvideo.io
from matplotlib import pyplot
from tqdm import tqdm
import flow_vis

this_filepath = Path(__file__)
this_filename = Path(__file__).stem
this_metric_name = this_filename[:-12]


class EndpointError:
    def __init__(self, frames_data: pandas.DataFrame, verbose_log: bool = True) -> None:
        super().__init__()
        self.frames_data = frames_data
        self.verbose_log = verbose_log
        return

    @staticmethod
    def compute_frame_error(gt_flow: numpy.ndarray, eval_flow: numpy.ndarray):
        error = gt_flow.astype('float') - eval_flow.astype('float')
        endpoint_error = numpy.mean(numpy.abs(error))
        return endpoint_error

    def compute_avg_error(self, old_data: pandas.DataFrame, database_dirpath: Path, pred_videos_dirpath: Path,
                          flow_folder_name: str):
        """

        :param old_data:
        :param database_dirpath: Should be path to Databases/VeedDynamic/Data
        :param pred_videos_dirpath:
        :param flow_folder_name:
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
            gt_flow_path = database_dirpath / f'all_short/RenderedData/{video_name}/seq{seq_num:02}/optical_flow_object_motion/{pred_frame_num-1:04}.exr'
            pred_flow_path = pred_videos_dirpath / f'{video_name}/seq{seq_num:02}/{flow_folder_name}/{pred_frame_num-1:04}.npz'
            gt_flow = self.read_flow(gt_flow_path)
            pred_flow = self.read_flow(pred_flow_path) * -0.5
            qa_score = self.compute_frame_error(gt_flow, pred_flow)
            qa_scores.append([video_name, seq_num, pred_frame_num, qa_score])
        qa_scores_data = pandas.DataFrame(qa_scores, columns=[
            'video_name', 'seq_num', 'pred_frame_num', this_metric_name])

        merged_data = self.update_qa_frame_data(old_data, qa_scores_data)
        merged_data = merged_data.round({this_metric_name: 4, })

        avg_error = numpy.mean(merged_data[this_metric_name])
        if isinstance(avg_error, numpy.ndarray):
            avg_error = avg_error.item()
        avg_error = numpy.round(avg_error, 4)
        return avg_error, merged_data

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

    @staticmethod
    def read_flow(path: Path):
        if path.suffix == '.npy':
            flow = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as flow_data:
                flow = flow_data[flow_data.files[0]]
        elif path.suffix == '.exr':
            exr_file = OpenEXR.InputFile(path.as_posix())
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x

            raw_bytes_b = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            vector_b = numpy.frombuffer(raw_bytes_b, dtype=numpy.float32)
            of_b = numpy.reshape(vector_b, (height, width))

            raw_bytes_a = exr_file.channel('A', Imath.PixelType(Imath.PixelType.FLOAT))
            vector_a = numpy.frombuffer(raw_bytes_a, dtype=numpy.float32)
            of_a = numpy.reshape(vector_a, (height, width))

            flow = numpy.stack([-of_b, of_a], axis=2)
        else:
            raise RuntimeError(f'Unknown flow format: {path.as_posix()}')

        return flow


# noinspection PyUnusedLocal
def start_qa(pred_videos_dirpath: Path, database_dirpath: Path, frames_datapath: Path, flow_folder_name: str):
    if not pred_videos_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: pred_videos_dirpath does not exist')
        return

    if not database_dirpath.exists():
        print(f'Skipping QA of folder: {pred_videos_dirpath.stem}. Reason: database_dirpath does not exist')
        return

    qa_scores_filepath = pred_videos_dirpath / 'QA_Scores.json'
    error_data_path = pred_videos_dirpath / f'QA_Scores/{flow_folder_name}/{this_metric_name}_FrameWise.csv'
    if qa_scores_filepath.exists():
        with open(qa_scores_filepath.as_posix(), 'r') as qa_scores_file:
            qa_scores = json.load(qa_scores_file)
    else:
        qa_scores = {}

    if flow_folder_name in qa_scores:
        if this_metric_name in qa_scores[flow_folder_name]:
            avg_error = qa_scores[flow_folder_name][this_metric_name]
            print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {flow_folder_name}: {avg_error}')
            return
    else:
        qa_scores[flow_folder_name] = {}

    if error_data_path.exists():
        error_data = pandas.read_csv(error_data_path)
    else:
        error_data = None

    frames_data = pandas.read_csv(frames_datapath)

    mse_computer = EndpointError(frames_data)
    avg_error, error_data = mse_computer.compute_avg_error(
        error_data, database_dirpath, pred_videos_dirpath, flow_folder_name)
    qa_scores[flow_folder_name][this_metric_name] = avg_error
    print(f'Average {this_metric_name}: {pred_videos_dirpath.as_posix()} - {flow_folder_name}: {avg_error}')
    with open(qa_scores_filepath.as_posix(), 'w') as qa_scores_file:
        simplejson.dump(qa_scores, qa_scores_file, indent=4)
    error_data_path.parent.mkdir(parents=True, exist_ok=True)
    error_data.to_csv(error_data_path, index=False)
    return


def demo1():
    pred_videos_dirpath = Path('../../../FlowEstimation/Research/004_HoleyFlowEstimation/Runs/Testing/Test0058_pred')
    # pred_videos_dirpath = Path('../../../FlowEstimation/Research/008_MpiFlowEstimation3d/Runs/Testing/Test0013')
    database_dirpath = Path('../../../../Databases/VeedDynamic/Data')
    frames_data_path = Path('../../../../Databases/VeedDynamic/Data/TrainTestSets/Set01/TestVideosData.csv')
    pred_folder_name = 'ObjectMotionFlow'
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
    if pred_folder_name is None:
        raise RuntimeError(f'Please provide pred_folder_name')

    start_qa(pred_videos_dirpath, database_dirpath, frames_datapath, pred_folder_name)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_function_name', default='demo1')
    parser.add_argument('--pred_videos_dirpath')
    parser.add_argument('--database_dirpath')
    parser.add_argument('--frames_datapath')
    parser.add_argument('--pred_folder_name')
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
