# Shree KRISHNAya Namaha
# Trainer for disocclusion infilling model
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import datetime
import json
import os
import random
from pathlib import Path

import flow_vis
import numpy
import pandas
import simplejson
import skimage.io
import skimage.transform
import torch
from deepdiff import DeepDiff
from matplotlib import pyplot
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import CommonUtils
from video_inpainting.data_loaders.DataLoaderFactory import get_data_loader
from video_inpainting.flow_predictors.FlowPredictorFactory import get_flow_predictor
from video_inpainting.frame_predictors.FramePredictorFactory import get_frame_predictor
from video_inpainting.loss_functions.LossComputer01 import LossComputer

this_filepath = Path(__file__)
this_filename = this_filepath.stem

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, configs: dict, train_num: int, train_dataset: torch.utils.data.Dataset,
                 val_dataset: torch.utils.data.Dataset, frame_predictor, loss_computer: LossComputer, optimizer,
                 output_dirpath: Path, verbose_log: bool = True):
        self.configs = configs
        self.train_num = train_num
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_data_loader = None
        self.val_data_loader = None
        self.train_data_iterator = None
        self.val_data_iterator = None
        self.frame_predictor = frame_predictor
        self.loss_computer = loss_computer
        self.optimizer = optimizer
        self.device = CommonUtils.get_device(self.configs['device'])
        self.output_dirpath = output_dirpath
        self.logger = SummaryWriter((output_dirpath / 'Logs').as_posix())
        self.verbose_log = verbose_log

        self.frame_predictor.to(self.device)
        return

    def train_one_iter(self, iter_num):
        def update_losses_dict_(epoch_losses_dict_: dict, iter_losses_dict_: dict, num_samples_: int):
            if epoch_losses_dict_ is None:
                epoch_losses_dict_ = {}
                for loss_name_ in iter_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] = iter_losses_dict_[loss_name_].item() * num_samples_
            else:
                for loss_name_ in epoch_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] += (iter_losses_dict_[loss_name_].item() * num_samples_)
            return epoch_losses_dict_

        def delete_dict(dict_data_: dict):
            for key_ in list(dict_data_.keys()):
                del dict_data_[key_]
            return

        iter_losses_dict = None
        total_num_samples = 0
        self.optimizer.zero_grad(set_to_none=True)
        for i in range(self.configs['batch_size'] // self.configs['sub_batch_size']):
            input_batch = self.get_next_train_batch()
            input_batch['iter_num'] = iter_num
            input_batch = CommonUtils.move_to_device(input_batch, self.device)
            output_batch = self.frame_predictor(input_batch)
            sub_iter_losses_dict = self.loss_computer.compute_losses(input_batch, output_batch)
            sub_iter_losses_dict['TotalLoss'].backward()

            # Update losses and the number of samples
            input_batch_first_element = input_batch[list(input_batch.keys())[0]]
            if isinstance(input_batch_first_element, torch.Tensor):
                batch_num_samples = input_batch_first_element.shape[0]
            elif isinstance(input_batch_first_element, list):
                batch_num_samples = len(input_batch_first_element)
            else:
                raise RuntimeError('Please help me! I don\'t know how to compute num_samples')
            iter_losses_dict = update_losses_dict_(iter_losses_dict, sub_iter_losses_dict, batch_num_samples)
            total_num_samples += batch_num_samples

            delete_dict(output_batch)
            delete_dict(input_batch)
            del output_batch, input_batch
        self.optimizer.step()

        for loss_name in iter_losses_dict.keys():
            iter_losses_dict[loss_name] = iter_losses_dict[loss_name] / total_num_samples

        return iter_losses_dict

    def run_validation(self, iter_num):
        def update_losses_dict_(epoch_losses_dict_: dict, iter_losses_dict_: dict, num_samples_: int):
            if epoch_losses_dict_ is None:
                epoch_losses_dict_ = {}
                for loss_name_ in iter_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] = iter_losses_dict_[loss_name_].item() * num_samples_
            else:
                for loss_name_ in epoch_losses_dict_.keys():
                    epoch_losses_dict_[loss_name_] += (iter_losses_dict_[loss_name_].item() * num_samples_)
            return epoch_losses_dict_

        epoch_losses_dict = None
        total_num_samples = 0
        self.frame_predictor.eval()
        with torch.no_grad():
            for i in range(self.configs['num_validation_iterations']):
                input_batch = self.get_next_val_batch()
                input_batch['iter_num'] = iter_num
                input_batch = CommonUtils.move_to_device(input_batch, self.device)
                output_batch = self.frame_predictor(input_batch)
                iter_losses_dict = self.loss_computer.compute_losses(input_batch, output_batch)

                # Update losses and the number of samples
                input_batch_first_element = input_batch[list(input_batch.keys())[0]]
                if isinstance(input_batch_first_element, torch.Tensor):
                    batch_num_samples = input_batch_first_element.shape[0]
                elif isinstance(input_batch_first_element, list):
                    batch_num_samples = len(input_batch_first_element)
                else:
                    raise RuntimeError('Please help me! I don\'t know how to compute num_samples')
                epoch_losses_dict = update_losses_dict_(epoch_losses_dict, iter_losses_dict, batch_num_samples)
                total_num_samples += batch_num_samples

        for loss_name in epoch_losses_dict.keys():
            epoch_losses_dict[loss_name] = epoch_losses_dict[loss_name] / total_num_samples
        self.frame_predictor.train()
        return epoch_losses_dict

    def get_next_train_batch(self):
        try:
            next_batch = next(self.train_data_iterator)
        except StopIteration:
            self.train_data_iterator = iter(self.train_data_loader)
            next_batch = next(self.train_data_iterator)
        return next_batch

    def get_next_val_batch(self):
        try:
            next_batch = next(self.val_data_iterator)
        except StopIteration:
            self.val_data_iterator = iter(self.val_data_loader)
            next_batch = next(self.val_data_iterator)
        return next_batch

    def train(self):
        def update_losses_data_(iter_num_: int, iter_losses_: dict, label: str):
            curr_time = datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p')
            self.logger.add_text(f'{label}/Time', curr_time, iter_num_)
            for key, value in iter_losses_.items():
                self.logger.add_scalar(f'{label}/{key}', value, iter_num_)
            return

        train_num = self.configs['train_num']
        print(f'Training {train_num} begins...')
        logs_dirpath = self.output_dirpath / 'Logs'
        sample_images_dirpath = self.output_dirpath / 'Samples'
        saved_models_dirpath = self.output_dirpath / 'SavedModels'
        logs_dirpath.mkdir(exist_ok=True)
        sample_images_dirpath.mkdir(exist_ok=True)
        saved_models_dirpath.mkdir(exist_ok=True)

        batch_size = self.configs['sub_batch_size']
        validation_interval = self.configs['validation_interval']
        sample_save_interval = self.configs['sample_save_interval']
        model_save_interval = self.configs['model_save_interval']
        total_num_iters = self.configs['num_iterations']

        self.train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True,
                                            pin_memory=True, num_workers=4, drop_last=False)
        self.val_data_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, num_workers=4, drop_last=False)
        self.train_data_iterator = iter(self.train_data_loader)
        self.val_data_iterator = iter(self.val_data_loader)
        start_iter_num = self.load_model(saved_models_dirpath)
        for iter_num in tqdm(range(start_iter_num, total_num_iters), initial=start_iter_num, total=total_num_iters,
                             mininterval=1, leave=self.verbose_log):
            iter_losses_dict = self.train_one_iter(iter_num)
            update_losses_data_(iter_num + 1, iter_losses_dict, 'train')

            if (iter_num + 1) % validation_interval == 0:
                epoch_val_loss = self.run_validation(iter_num)
                update_losses_data_(iter_num + 1, epoch_val_loss, 'validation')

            if (iter_num + 1) % sample_save_interval == 0:
                self.save_sample_images(iter_num + 1, sample_images_dirpath)

            if (iter_num + 1) % model_save_interval == 0:
                self.save_model(iter_num + 1, saved_models_dirpath)

        save_plots(logs_dirpath)
        return

    def save_sample_images(self, iter_num, save_dirpath):
        resolution = self.configs['data_loader']['patch_size']

        def convert_tensor_to_image_(tensor_batch_):
            np_array = tensor_batch_.detach().cpu().numpy()
            image_batch = (numpy.moveaxis(np_array, [0, 1, 2, 3], [0, 3, 1, 2]))
            if image_batch.shape[3] == 2:
                # Convert flow to image
                image_batch = numpy.stack([flow_vis.flow_to_color(flow) for flow in image_batch])
            resized_batch = numpy.stack([skimage.transform.resize(image, resolution) for image in image_batch])
            resized_batch = numpy.round(resized_batch * 255).astype('uint8')
            return resized_batch

        def create_collage_(input_batch_, output_batch_):
            frame1 = input_batch_['frame'][:4]
            pred_frame = output_batch_['pred_frame'][:4]
            target_frame = input_batch_['target_frame'][:4]
            sample_images = [frame1, pred_frame, target_frame]

            for i in range(len(sample_images)):
                # noinspection PyTypeChecker
                numpy_batch = convert_tensor_to_image_(sample_images[i])
                padded = numpy.pad(numpy_batch, ((0, 0), (5, 5), (5, 5), (0, 0)), mode='constant', constant_values=255)
                sample_images[i] = numpy.concatenate(padded, axis=0)
            sample_collage_ = numpy.concatenate(sample_images, axis=1)
            return sample_collage_

        self.frame_predictor.eval()
        with torch.no_grad():
            # train set samples
            train_input_batch = self.get_next_train_batch()
            train_input_batch = CommonUtils.move_to_device(train_input_batch, self.device)
            train_output_batch = self.frame_predictor(train_input_batch)
            train_sample_collage = create_collage_(train_input_batch, train_output_batch)

            # validation set samples
            val_input_batch = self.get_next_val_batch()
            val_input_batch = CommonUtils.move_to_device(val_input_batch, self.device)
            val_output_batch = self.frame_predictor(val_input_batch)
            val_sample_collage = create_collage_(val_input_batch, val_output_batch)
        self.frame_predictor.train()

        save_path = save_dirpath / f'Iter{iter_num:06}.png'
        sample_collage = numpy.concatenate([train_sample_collage, val_sample_collage], axis=0)
        skimage.io.imsave(save_path.as_posix(), sample_collage)
        return

    def save_model(self, iter_num: int, save_dirpath: Path, label: str = None):
        if label is None:
            label = f'Iter{iter_num:06}'
        save_path1 = save_dirpath / f'Model_{label}.tar'
        save_path2 = save_dirpath / f'Model_Latest.tar'
        checkpoint_state = {
            'iteration_num': iter_num,
            'model_state_dict': self.frame_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint_state, save_path1)
        if save_path2.exists():
            os.remove(save_path2)
        os.system(f'ln -s {os.path.relpath(save_path1, save_path2.parent)} {save_path2.as_posix()}')
        return

    def load_model(self, saved_models_dirpath: Path):
        latest_model_path = saved_models_dirpath / 'Model_Latest.tar'
        if latest_model_path.exists():
            if self.device.type == 'cpu':
                checkpoint_state = torch.load(latest_model_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(latest_model_path, map_location=self.device)
            iter_num = checkpoint_state['iteration_num']
            self.frame_predictor.load_state_dict(checkpoint_state['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
            print(f'Resuming Training from iteration {iter_num + 1}')
        else:
            iter_num = 0
        return iter_num


def save_plots(logs_dirpath: Path):
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(logs_dirpath.as_posix(), size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    for loss_key in ea.scalars.Keys():
        prefix, loss_name = loss_key.split('/')
        loss_data = pandas.DataFrame(ea.Scalars(loss_key))
        iter_nums = loss_data['step'].to_numpy()
        loss_values = loss_data['value'].to_numpy()
        save_path = logs_dirpath / f'{prefix}_{loss_name}.png'
        pyplot.plot(iter_nums, loss_values)
        pyplot.savefig(save_path)
        pyplot.close()
    return


def init_seeds(seed: int = 1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    return


def save_configs(output_dirpath: Path, configs: dict):
    # Save configs
    configs_path = output_dirpath / 'Configs.json'
    if configs_path.exists():
        # If resume_training is false, an error would've been raised when creating output directory. No need to handle
        # it here.
        with open(configs_path.as_posix(), 'r') as configs_file:
            old_configs = json.load(configs_file)
        configs['seed'] = old_configs['seed']
        for key in old_configs.keys():
            if key not in configs.keys():
                configs[key] = old_configs[key]
        if configs['num_iterations'] > old_configs['num_iterations']:
            old_configs['num_iterations'] = configs['num_iterations']
        old_configs['device'] = configs['device']
        if configs != old_configs:
            raise RuntimeError(f'Configs mismatch while resuming training: {DeepDiff(old_configs, configs)}')
    with open(configs_path.as_posix(), 'w') as configs_file:
        simplejson.dump(configs, configs_file, indent=4)
    return


def start_training(configs: dict):
    root_dirpath = Path('../../')

    # Setup output dirpath
    train_num = configs['train_num']
    output_dirpath = root_dirpath / f'Runs/Training/Train{train_num:04}'
    output_dirpath.mkdir(parents=True, exist_ok=configs['resume_training'])
    save_configs(output_dirpath, configs)
    init_seeds(configs['seed'])

    # Create data_loaders, models, optimizers etc
    configs['root_dirpath'] = root_dirpath
    database_dirpath = root_dirpath / f'Data/Databases/{configs["database_name"]}'
    set_num = configs['video_inpainting']['data_loader']['train_set_num']
    train_videos_datapath = root_dirpath / f'res/TrainTestSets/{configs["database_name"]}/Set{set_num:02}/TrainVideosData.csv'
    val_videos_datapath = root_dirpath / f'res/TrainTestSets/{configs["database_name"]}/Set{set_num:02}/ValidationVideosData.csv'
    train_dataset = get_data_loader(configs, database_dirpath, train_videos_datapath)
    val_dataset = get_data_loader(configs, database_dirpath, val_videos_datapath)
    flow_predictor = get_flow_predictor(configs)
    frame_predictor = get_frame_predictor(configs, flow_predictor)
    loss_computer = LossComputer(configs)
    optimizer = torch.optim.Adam(list(frame_predictor.parameters()), lr=configs['lr'],
                                 betas=(configs['beta1'], configs['beta2']))

    # Start training
    trainer = Trainer(configs, train_num, train_dataset, val_dataset, frame_predictor, loss_computer, optimizer,
                      output_dirpath)
    trainer.train()

    del trainer, optimizer, loss_computer, frame_predictor, flow_predictor
    torch.cuda.empty_cache()
    return
