#!/bin/bash
set -x

config_file=$1 # config/deformable_detr_r50.yaml
weights=$2 # assets/model_zoo/r50_deformable_detr-checkpoint_d2.pth
output_dir=$3 # saved/eval
gpus=$4 # 1

export DETECTRON2_DATASETS=YOUR_DATA_PATH
python train_net.py \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    --eval-only \
    --config-file $config_file \
    --num-gpus $gpus \
    MODEL.WEIGHTS $weights \
    OUTPUT_DIR $output_dir