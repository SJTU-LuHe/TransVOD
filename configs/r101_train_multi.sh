#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/multibaseline/r101_grad/e7_nf1_ld4,6_lr0.0002_nq300_wbox_MEGA_detrNorm_preSingle_nr14_dc5_nql3_filter150_75_40
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet101 \
    --epochs 7 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume exps/r101checkpoint0009.pth \
    --lr_drop_epochs 4 6 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vid_multi \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
