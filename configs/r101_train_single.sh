#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/singlebaseline/r101_e8_nf4_ld6,7_lr0.0002_nq300_bs4_wbox_joint_MEGA_detrNorm_class31_pretrain_coco_dc5
mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet101 \
    --epochs 10 \
    --num_feature_levels 1\
    --num_queries 300 \
    --dilation \
    --batch_size 4 \
    --num_workers 8 \
    --resume ./exps/our_models/COCO_pretrained_model/r101_deformable_detr_single_scale_bbox_refinement-dc5_checkpoint0049.pth \
    --lr_drop_epochs 7 9 \
    --with_box_refine \
    --coco_pretrain \
    --dataset_file vid_single \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
     # --resume /mnt/lustre/helu/code/vod/video_object_detection/exps/pretrainModel/r50_deformable_detr_single_scale_dc5-checkpoint.pth \
