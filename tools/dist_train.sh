#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

# $PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}

$PYTHON -m torch.distributed.launch --nproc_per_node=8 --nnodes=3 --node_rank=$1 --master_addr="10.141.8.84" --master_port=9876 tools/train.py configs/traffic_htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py --launcher pytorch
