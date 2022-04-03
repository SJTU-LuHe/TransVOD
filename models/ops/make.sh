#!/usr/bi/nenv bash
export PATH=/mnt/lustre/share/gcc/gcc-5.4/bin/:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:/mnt/lustre/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST='3.5;5.0+PTX;6.0;7.0'

partition=$1
srun -p $partition --gres=gpu:1 python setup.py build develop --user

# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

