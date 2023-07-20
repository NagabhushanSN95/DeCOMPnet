# DeCOMPnet
Official code release for the ISMAR 2022 paper "Temporal View Synthesis of Dynamic Scenes through 3D Object Motion Estimation with Multi-Plane Images"

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-view-synthesis-of-dynamic-scenes/temporal-view-synthesis-on-iisc-veed-dynamic)](https://paperswithcode.com/sota/temporal-view-synthesis-on-iisc-veed-dynamic?p=temporal-view-synthesis-of-dynamic-scenes)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-view-synthesis-of-dynamic-scenes/temporal-view-synthesis-on-mpi-sintel)](https://paperswithcode.com/sota/temporal-view-synthesis-on-mpi-sintel?p=temporal-view-synthesis-of-dynamic-scenes)

## Databases
* Download the [IISc VEED-Dynamic database](https://nagabhushansn95.github.io/publications/2022/DeCOMPnet.html#database-download). Merge all the data and place it in `/Data/Databases/VeedDynamic/all_short`. 
* For MPI-Sintel database, download all the ground truth for the training set from [here](http://sintel.is.tue.mpg.de/downloads). Use the scripts in `src/utils/mpi_sintel` to extract the required data and to organize it. The following steps describe training and testing on IISc VEED-Dynamic dataset. The steps for MPI-Sintel dataset are similar and the code for each step is also provided. 
* For other datasets, follow similar procedure.

## Python Environment
Environment details are available in `EnvironmentData/DeCOMPnet.yml`. The environment can be created using conda
```shell
cd EnvironmentData
conda env create -f DeCOMPnet.yml
cd ..
```

## Training and Inference
DeCOMPnet uses a two-stage training procedure, one to train MPI flow estimation model and the other to train infilling model. Follow the below steps to train both the models

1. Generate necessary data for training flow estimation model - warped frame to nullify camera motion and a masks to indicating regions containing non-zero flow.
```shell
cd data_generators
python ObjectMotionIsolation01_VeedDynamic.py
python LOF_POI_01_VeedDynamic.py
cd ..
```

2. Download [pre-trained ARFlow weights](https://github.com/lliuz/ARFlow/tree/master/checkpoints) and place them in `PretrainedModels/ARFlow`

3. Convert ARFlow weights to convention used in this repository
```shell
cd flow_estimation/utils
python ARFlowWeightsConverter01.py
cd ../..
```

4. Train the flow estimation model
```shell
cd flow_estimation
python VeedDynamicTrainer01.py
cd ..
```

5. Estimate local flow between past frames and generate motion warped frames
```shell
cd data_generators
python LocalOpticalFlow01_VeedDynamic.py
python MotionWarping01_VeedDynamic.py
```

6. Train the disocclusion infilling model
```shell
cd video_inpainting
python VeedDynamicTrainer01.py
cd ..
```

7. To run ST-RRED, download [code from here](https://github.com/utlive/strred/tree/main/matlabPyrTools) and place it in `src/qa/05_CroppedSTRRED/src/matlab`. If you want to skip computing ST-RRED, comment the corresponding line in `src/qa/00_Common/src/AllMetrics01_VeedDynamic.py`.

8. Test the model on the IISc VEED-Dynamic dataset and run QA
```shell
python VeedDynamicTester01.py
```

## Pretrained Weights
Pretrained weights of the flow estimation and disocclusion infilling are available [here](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/nagabhushans_iisc_ac_in/ErZCZr5Ky9BBtaBI82Xq0sABX3h5EkcKkf6SwAGMxQDEAg?e=pthZHd). 
Download the weights and provide the corresponding paths in the [Tester01.py](src/Tester01.py#L253-L255).

If you use DeCOMPnet model in your publication, please specify the version as well. The current version is 1.0.

## License
MIT License

Copyright (c) 2022 Nagabhushan Somraj, Rajiv Soundararajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Citation
If you use this code for your research, please cite our paper

```bibtex
@article{somraj2022DeCOMPnet,
    title = {Temporal View Synthesis of Dynamic Scenes through 3D Object Motion Estimation with Multi-Plane Images},
    author = {Somraj, Nagabhushan and Sancheti, Pranali and Soundararajan, Rajiv},
    booktitle = {Proceedings of the IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
    pages={817-826},
    year = {2022},
    doi = {10.1109/ISMAR55827.2022.00100}
}
```

## Acknowledgements
The code and initialization weights for optical flow estimation is borrowed from [ARFlow](https://github.com/lliuz/ARFlow). However, significant changes have been made on top of the code and so the code might look different. The code for camera motion warping is borrowed from [here](https://github.com/NagabhushanSN95/Pose-Warping).


For any queries or bugs related to either DeCOMPnet code or IISc VEED-Dynamic database, please raise an issue.