# Shree KRISHNAya Namaha
# For every video in all folder, number of frames is detected and saved in a csv file
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import time
import datetime
import traceback
import numpy
import skimage.io
import skvideo.io
import pandas
import simplejson

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def detect_num_frames(all_dirpath: Path) -> pandas.DataFrame:
    num_frames_data = []
    for video_dirpath in sorted(all_dirpath.iterdir()):
        clean_dirpath = video_dirpath / 'rgb/clean'
        num_frames = len(list(clean_dirpath.iterdir()))
        num_frames_data.append([video_dirpath.stem, num_frames])
    num_frames_data = pandas.DataFrame(num_frames_data, columns=['video_name', 'num_frames'])
    return num_frames_data


def demo1():
    all_dirpath = Path('../Data/ExtractedData/all')
    output_path = Path('../Data/ExtractedData/NumberOfFramesPerVideo.csv')

    num_frames_data = detect_num_frames(all_dirpath)
    num_frames_data.to_csv(output_path, index=False)
    return


def main():
    demo1()
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
