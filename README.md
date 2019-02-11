# Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction

## Introduction

This is a PyTorch implementation of our WACV 2018 paper "[`Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction`](https://arxiv.org/pdf/1801.03986.pdf)".

![Alt Text](demo/Movie_20140401_03_033.gif)

***Note:*** The pretrained models are trained on the [`split1`](./data/ice.json) of following larger dataset.

## Environment

- The code is developed with CUDA 9.0, ***Python >= 3.6***, ***PyTorch >= 1.0***

## Data Preparation

1. Download the raw data at `ftp://data.cresis.ku.edu/data/rds/2014_Greenland_P3/CSARP_music3D/`

    - If you don't want to preprocess the data by yourself, please use [`create_slices.m`](./scripts/create_slices_64x64/create_slices.m) to generate radar images and [`convert_mat_to_npy.py`](./scripts/convert_mat_to_npy.py) to convert them from MATLAB to NumPy files.

2. Make sure to put the files as the following structure:
    ```
    YOUR_PATH_TO_CRESIS_DATASET
    ├── slices_mat_64x64
    |   ├── 20140325_05
    |   |   ├── 001
    |   |   |   ├── 00001.mat
    |   |   |   ├── ...
    |   |   ├── ...
    │   ├── ...
    |
    ├── slices_npy_64x64
    |   ├── 20140325_05
    |   |   ├── 001
    |   |   |   ├── 00001.npy
    |   |   |   ├── ...
    |   |   ├── ...
    |   ├── ...
    ```

3. Create softlinks of datasets:
    ```
    cd ice-wacv2018
    ln -s YOUR_PATH_TO_CRESIS_DATASET data/CReSIS
    ln -s data/target data/CReSIS/target
    ```

## Pretrained Models

- Download the pretrained model at [`model_zoo`](./model_zoo).

## Training

- C3D
```
cd ice-wacv2018
# Default Hyperparameters
python tools/c3d/train.py
# OR
python tools/c3d/train.py --gpu XXX --batch_size XXX --lr XXX
```

- Extract C3D Features
```
cd ice-wacv2018
# Default Hyperparameters and Paths
python tools/c3d/extract_features.py
# OR
python tools/c3d/extract_features.py --gpu XXX --checkpoint YOUR_PATH_TO_C3D --batch_size XXX
```

- RNN
```
cd ice-wacv2018
# Default Hyperparameters
python tools/rnn/train.py
# OR
python tools/rnn/train.py --gpu XXX --batch_size XXX --lr XXX
```

## Evaluation
```
cd ice-wacv2018
# Default Hyperparameters and Paths
python e2e_run.py
# OR
python e2e_run.py --gpu XXX --data_root YOUR_PATH_TO_CRESIS_DATASET --c3d_pth YOUR_PATH_TO_C3D --rnn_pth YOUR_PATH_TO_RNN
```

## Citations

If you are using the data/code/model provided here in a publication, please cite our papers:

    @inproceedings{ice2018wacv, 
        title = {Multi-Task Spatiotemporal Neural Networks for Structured Surface Reconstruction},
        author = {Mingze Xu and Chenyou Fan and John Paden and Geoffrey Fox and David Crandall},
        booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
        year = {2018},
    }

    @inproceedings{icesurface2017icip, 
        title = {Automatic estimation of ice bottom surfaces from radar imagery},
        author = {Mingze Xu and David J. Crandall and Geoffrey C. Fox and John D. Paden},
        booktitle = {IEEE International Conference on Image Processing (ICIP)},
        year = {2017},
    }
