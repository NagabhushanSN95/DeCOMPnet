# Shree KRISHNAya Namaha
# Trains on IISc VEED-Dynamic Database
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import time
import traceback
from pathlib import Path

import numpy

import Trainer01 as Trainer

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def demo1():
    train_num = 2

    train_configs = {
        'trainer': f'{this_filename}/{Trainer.this_filename}',
        'train_num': train_num,
        'database_name': 'VeedDynamic',
        'video_inpainting': {
            'data_loader': {
                'name': 'VeedDynamic01',
                'patch_size': [256, 256],
                'train_set_num': 1,
                'num_mpi_planes': 4,
                'local_global_dirname': 'LGMP_01_1step_forward',
            },
            'flow_predictor': {
                'name': 'Unet01',
            },
            'frame_predictor': {
                'name': 'FramePredictor01',
                'num_iterations': 3,
            },
            'losses': [
                {
                    'name': 'MSE01',
                    'weight': 1,
                },
            ],
        },
        'resume_training': True,
        'num_iterations': 10000,
        'batch_size': 4,
        'sub_batch_size': 4,
        'lr': 0.0001,
        'beta1': 0.9,
        'beta2': 0.999,
        'validation_interval': 100,
        'num_validation_iterations': 10,
        'sample_save_interval': 50,
        'model_save_interval': 500,
        'seed': numpy.random.randint(1000),
        'device': 'gpu0',
    }
    Trainer.start_training(train_configs)
    return


def demo3():
    """
    Saves plots mid training
    :return:
    """
    train_num = 1
    loss_plots_dirpath = Path(f'../Runs/Training/Train{train_num:04}/Logs')
    Trainer.save_plots(loss_plots_dirpath)
    import sys
    sys.exit(0)


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
        run_result = 'Error: ' + str(e)
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
