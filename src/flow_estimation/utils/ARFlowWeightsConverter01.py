# Shree KRISHNAya Namaha
# Converts ARFlow weights to be suitable with the naming convention used here
# Author: Nagabhushan S N
# Last Modified: 26/08/2022

import time
import datetime
import traceback

from pathlib import Path

import torch
from collections import OrderedDict

from flow_estimation.feature_extractors.FeatureExtractorFactory import get_feature_extractor
from flow_estimation.flow_estimators.FlowEstimatorFactory import get_flow_estimator
from flow_estimation.frame_predictors.FramePredictorFactory import get_frame_predictor
from utils import CommonUtils

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def convert_weights(arflow_weights_path: Path, model_configs: dict, save_path: Path):
    device = CommonUtils.get_device(model_configs['device'])
    search_range_spatial = model_configs['flow_estimation']['flow_estimator']['search_range_spatial']
    search_range_depth = model_configs['flow_estimation']['flow_estimator']['search_range_depth']

    feature_extractor = get_feature_extractor(model_configs)
    flow_estimator = get_flow_estimator(model_configs)
    frame_predictor = get_frame_predictor(model_configs, feature_extractor, flow_estimator)

    checkpoint = torch.load(arflow_weights_path.as_posix(), map_location=device)
    if 'state_dict' in checkpoint:
        old_state_dict = (checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        old_state_dict = (checkpoint['model_state_dict'])
    else:
        old_state_dict = checkpoint

    # Rename weights
    model_keys = set(frame_predictor.state_dict().keys())
    weight_keys = set(old_state_dict.keys())
    new_state_dict = OrderedDict()
    for old_key in old_state_dict.keys():
        new_key = ''
        if 'feature_pyramid_extractor' in old_key:
            new_key += 'feature_extractor'
            conv_num = int(old_key[32]) + 1
            new_key += f'.conv{conv_num}'
            if old_key[34] == '0':
                new_key += 'a'
            elif old_key[34] == '1':
                new_key += 'b'
            else:
                raise RuntimeError
            new_key += old_key[37:]
        elif 'conv_1x1' in old_key:
            new_key += 'flow_estimator.feature_aggregator'
            conv_num = int(old_key[9]) + 1
            new_key += f'.conv{conv_num}'
            new_key += old_key[12:]
        elif 'flow_estimators' in old_key:
            new_key += 'flow_estimator.crude_flow_estimator'
            if 'predict_flow' in old_key:
                new_key += '.conv6'
                new_key += old_key[30:]
            else:
                conv_num = int(old_key[20])
                new_key += f'.conv{conv_num}'
                new_key += old_key[23:]
        elif 'context_networks' in old_key:
            new_key += 'flow_estimator.fine_flow_estimator'
            conv_num = int(old_key[23]) + 1
            new_key += f'.conv{conv_num}'
            new_key += old_key[26:]
        else:
            print(old_key)
            raise RuntimeError
        # assert old_state_dict[old_key].shape == frame_predictor.state_dict()[new_key].shape
        new_state_dict[new_key] = old_state_dict[old_key]

    # Convert 2D weights to 3D weights
    old_model_dict = new_state_dict
    del new_state_dict
    new_model_dict = OrderedDict()
    for key in old_model_dict.keys():
        # if key.startswith('feature_extractor.conv1a') and key.endswith('weight'):
        #     old_weight = old_model_dict[key]
        #     zeros = torch.zeros_like(old_weight)
        #     new_weight = torch.stack([zeros, old_weight, zeros], dim=4)
        #     zeros = torch.zeros_like(new_weight[:, :1])
        #     new_weight = torch.cat([new_weight, zeros], dim=1)
        #     new_model_dict[key] = new_weight
        if key.startswith('feature_extractor.conv') and key.endswith('weight'):
            old_weight = old_model_dict[key]
            zeros = torch.zeros_like(old_weight)
            new_weight = torch.stack([zeros, old_weight, zeros], dim=4)
            new_model_dict[key] = new_weight
        elif key.startswith('flow_estimator.feature_aggregator.conv') and key.endswith('weight'):
            old_weight = old_model_dict[key]
            new_weight = old_weight[:, :, :, :, None]
            new_model_dict[key] = new_weight
        elif key.startswith('flow_estimator.crude_flow_estimator.conv1') and key.endswith('weight'):
            old_weight = old_model_dict[key]
            spatial_corr_dim = (search_range_spatial * 2 + 1) ** 2
            depth_corr_dim = search_range_depth * 2 + 1
            zeros_s = torch.zeros([old_weight.shape[0], spatial_corr_dim * search_range_depth, *old_weight.shape[2:]]).to(old_weight)
            # zeros_d = torch.zeros_like(old_weight[:, :num_mpi_planes])  # PWCnet01
            zeros_d = torch.zeros_like(old_weight[:, :depth_corr_dim])  # PWCnet02
            new_weight = torch.cat([
                zeros_s, old_weight[:, :spatial_corr_dim], zeros_s, old_weight[:, spatial_corr_dim:], zeros_d
            ], dim=1)
            zeros = torch.zeros_like(new_weight)
            new_weight = torch.stack([zeros, new_weight, zeros], dim=4)
            new_model_dict[key] = new_weight
        elif key.startswith('flow_estimator.crude_flow_estimator.conv6') and key.endswith('weight'):
            old_weight = old_model_dict[key]
            zeros = torch.zeros_like(old_weight)
            new_weight = torch.stack([zeros, old_weight, zeros], dim=4)
            depth_corr_dim = search_range_depth * 2 + 1
            zeros_d = torch.zeros((depth_corr_dim, *new_weight.shape[1:])).to(old_weight)
            new_weight = torch.cat([new_weight, zeros_d], dim=0)
            new_model_dict[key] = new_weight
        elif key.startswith('flow_estimator.crude_flow_estimator.conv6') and key.endswith('bias'):
            old_weight = old_model_dict[key]
            depth_corr_dim = search_range_depth * 2 + 1
            zeros_d = torch.zeros((depth_corr_dim,)).to(old_weight)
            zeros_d[depth_corr_dim // 2] = 100
            new_weight = torch.cat([old_weight, zeros_d], dim=0)
            new_model_dict[key] = new_weight
        elif key.startswith('flow_estimator.crude_flow_estimator.conv') and key.endswith('weight'):
            old_weight = old_model_dict[key]
            zeros = torch.zeros_like(old_weight)
            new_weight = torch.stack([zeros, old_weight, zeros], dim=4)
            new_model_dict[key] = new_weight
        elif key.startswith('flow_estimator.fine_flow_estimator.conv'):
            pass
        else:
            new_model_dict[key] = old_model_dict[key]

    frame_predictor.load_state_dict(new_model_dict, strict=False)

    if 'state_dict' in checkpoint:
        del checkpoint['state_dict']
    if 'model_state_dict' in checkpoint:
        del checkpoint['model_state_dict']
    checkpoint['model_state_dict'] = new_model_dict
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    return


def demo1():
    old_weights_path = Path('../../../PretrainedModels/ARFlow/Sintel/pwclite_ar.tar')
    new_weights_path = Path('../../../PretrainedModels/ARFlow/Sintel/pwclite_ar_renamed.tar')
    model_configs = {
        'flow_estimation': {
            'data_loader': {
                'num_mpi_planes': 4,
            },
            'feature_extractor': {
                'name': 'PyramidFeatureExtractor01',
            },
            'flow_estimator': {
                'name': 'PWCnet01',
                'upsample_flow': True,
                'search_range_spatial': 4,
                'search_range_depth': 1,
            },
            'frame_predictor': {
                'name': 'FramePredictor01',
                'bidirectional_flow': False,
                'occlusion_mask': True,
            },
        },
        'batch_size': 2,
        'device': 'cpu',
    }
    convert_weights(old_weights_path, model_configs, new_weights_path)
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
