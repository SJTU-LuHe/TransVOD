#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/our_models/exps_single/r50
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --epochs 7 \
    --eval \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --resume ${EXP_DIR}/checkpoint0009.pth \
    --lr_drop_epochs 4 6 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vid_single \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
