# TransVOD:End-to-End Video Object Detection with Spatial-Temporal Transformers



This repository is an official implementation of the paper [TransVOD:End-to-End Video Object Detection with Spatial-Temporal Transformers](https://dlnext.acm.org/doi/10.1145/3474085.3475285).

## Introduction

**TransVOD**  is a fully end-to-end video object dectection framework based on Transformer. It directly outputs the detection results without any complicated post-processing methods.

<div style="align: center">
<img src=./figs/teaser.png/>
</div>

**Abstract.** 
Recently, DETR and Deformable DETR have been proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance as previous complex hand-crafted detectors. However, their performance on Video Object Detection (VOD) has not been well explored. In this paper, we present TransVOD, an end-to-end video object detection model based on a spatial-temporal Transformer architecture. The goal of this paper is to streamline the pipeline of VOD, effectively removing the need for many hand-crafted components for feature aggregation, e.g., optical flow, recurrent neural networks, relation networks. Besides, benefited from the object query design in DETR, our method does not need complicated post-processing methods such as Seq-NMS or Tubelet rescoring, which keeps the pipeline simple and clean. In particular, we present temporal Transformer to aggregate both the spatial object queries and the feature memories of each frame. Our temporal Transformer consists of three components: Temporal Deformable Transformer Encoder (TDTE) to encode the multiple frame spatial details, Temporal Query Encoder (TQE) to fuse object queries, and Temporal Deformable Transformer Decoder (TDTD) to obtain current frame detection results. These designs boost the strong baseline deformable DETR by a significant margin (3%-4% mAP) on the ImageNet VID dataset. TransVOD yields comparable results performance on the benchmark of ImageNet VID. We hope our TransVOD can provide a new perspective for video object detection.

## Updates
- (2022/04/03) Code and pretrained weights for TransVOD released. 


## Main Results

| **Method** | **Backbone** | **Frame Numbers** | **AP50**  |                                           **URL**                                           |
| :--------: | :---------: | :------------: | :------: | :-----------------------------------------------------------------------------------------: |
|    Deformable DETR   | ResNet50  |   1   |   76   |[model](https://drive.google.com/drive/folders/1FTRz-O1_-IL_la-2jQzDiZgvI_NLRPme?usp=sharing) |
|    Deformable DETR   | ResNet101  |   1   |   78.3   |[model](https://drive.google.com/drive/folders/1FTRz-O1_-IL_la-2jQzDiZgvI_NLRPme?usp=sharing) |
|    TransVOD   | ResNet50  |   15   |   79.9   |[model](https://drive.google.com/drive/folders/1FTRz-O1_-IL_la-2jQzDiZgvI_NLRPme?usp=sharing) |
|    TransVOD   | ResNet101  |   15   |   81.9   |[model](https://drive.google.com/drive/folders/1FTRz-O1_-IL_la-2jQzDiZgvI_NLRPme?usp=sharing) |



*Note:*
1. All models of TransVOD are trained  with pre-trained weights on COCO dataset.


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n TransVOD python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate TransVOD
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/)

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

1. Please download ILSVRC2015 DET and ILSVRC2015 VID dataset from [here](https://image-net.org/challenges/LSVRC/2015/2015-downloads). Then we covert jsons of two datasets by using the [code](https://github.com/open-mmlab/mmtracking/blob/master/tools/convert_datasets/ilsvrc/). The joint [json](https://drive.google.com/drive/folders/1cCXY41IFsLT-P06xlPAGptG7sc-zmGKF?usp=sharing)  of two datasets is provided. The  After that, we recommend to symlink the path to the datasets to datasets/. And the path structure should be as follows:

```
code_root/
└── data/
    └── vid/
        ├── Data
            ├── VID/
            └── DET/
        └── annotations/
        	  ├── imagenet_vid_train.json
            ├── imagenet_vid_train_joint_30.json
        	  └── imagenet_vid_val.json

```

### Training
We use ResNet50 and ResNet101 as the network backbone. We train our TransVOD with ResNet50 as backbone as following:

#### Training on single node
1. Train SingleBaseline. You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). 
   
```bash 
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 r50 $2 configs/r50_train_single.sh
```  
1. Train TransVOD. Using the model weights of SingleBaseline as the resume model.

```bash 
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 r50 $2 configs/r50_train_multi.sh
``` 


#### Training on slurm cluster
If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> r50 8 configs/r50_train_multi.sh
```

### Evaluation
You can get the config file and pretrained model of TransVOD (the link is in "Main Results" session), then put the pretrained_model into correponding folder.
```
code_root/
└── exps/
    └── our_models/
        ├── COCO_pretrained_model
        ├── exps_single
        └── exps_multi
```
And then run following command to evaluate it on ImageNET VID validation set:
```bash 
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 eval_r50 $2 configs/r50_eval_multi.sh
```



## Citing TransVOD
If you find TransVOD useful in your research, please consider citing:
```bibtex
@inproceedings{he2021end,
  title={End-to-End Video Object Detection with Spatial-Temporal Transformers},
  author={He, Lu and Zhou, Qianyu and Li, Xiangtai and Niu, Li and Cheng, Guangliang and Li, Xiao and Liu, Wenxuan and Tong, Yunhai and Ma, Lizhuang and Zhang, Liqing},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={1507--1516},
  year={2021}
}
@article{zhou2022transvod,
  title={TransVOD: End-to-end Video Object Detection with Spatial-Temporal Transformers},
  author={Zhou, Qianyu and Li, Xiangtai and He, Lu and Yang, Yibo and Cheng, Guangliang and Tong, Yunhai and Ma, Lizhuang and Tao, Dacheng},
  journal={arXiv preprint arXiv:2201.05047},
  year={2022}
}
