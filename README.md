# SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation

<img align="center" src="figures/animation.gif" width="750">

[Video](https://www.youtube.com/watch?v=pvM_bASOQmo)

This repository is the official implementation of our paper [SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation](https://arxiv.org/pdf/2002.10111.pdf).
For more details, please see our paper.

## Introduction
SMOKE is a **real-time** monocular 3D object detector for autonomous driving. 
The runtime on a single NVIDIA TITAN XP GPU is **~30ms**. 
Part of the code comes from [CenterNet](https://github.com/xingyizhou/CenterNet), 
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
and [Detectron2](https://github.com/facebookresearch/detectron2).

The performance on KITTI 3D detection (3D/BEV) is as follows:

|             |     Easy      |    Moderate    |     Hard     |
|-------------|:-------------:|:--------------:|:------------:|
| Car         | 14.17 / 21.08 | 9.88 / 15.13   | 8.63 / 12.91 | 
| Pedestrian  | 5.16  / 6.22  | 3.24 / 4.05    | 2.53 / 3.38  | 
| Cyclist     | 1.11  / 1.62  | 0.60 / 0.98    | 0.47 / 0.74  |

The pretrained weights can be downloaded [here](https://drive.google.com/open?id=11VK8_HfR7t0wm-6dCNP5KS3Vh-Qm686-).

## Requirements
All codes are tested under the following environment:
*   Ubuntu 16.04
*   Python 3.7
*   Pytorch 1.3.1
*   CUDA 10.0

## Dataset
We train and test our model on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Please first download the dataset and organize it as following structure:
```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```  

## Setup
1. We use `conda` to manage the environment:
```
conda create -n SMOKE python=3.7
```

2. Clone this repo:
```
git clone https://github.com/lzccccc/SMOKE
```

3. Build codes:
```
python setup.py build develop
```

4. Link to dataset directory:
```
mkdir datasets
ln -s /path_to_kitti_dataset datasets/kitti
```

## Getting started
First check the config file under `configs/`. 

We train the model on 4 GPUs with 32 batch size:
```
python tools/plain_train_net.py --num-gpus 4 --config-file "configs/smoke_gn_vector.yaml"
```

For single GPU training, simply run:
```
python tools/plain_train_net.py --config-file "configs/smoke_gn_vector.yaml"
```

We currently only support single GPU testing:
```
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"
```

## Acknowledgement
[CenterNet](https://github.com/xingyizhou/CenterNet)

[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

[Detectron2](https://github.com/facebookresearch/detectron2)


## Citations
Please cite our paper if you find SMOKE is helpful for your research.
```
@article{liu2020SMOKE,
  title={{SMOKE}: Single-Stage Monocular 3D Object Detection via Keypoint Estimation},
  author={Zechen Liu and Zizhang Wu and Roland T\'oth},
  journal={arXiv preprint arXiv:2002.10111},
  year={2020}
}
```
