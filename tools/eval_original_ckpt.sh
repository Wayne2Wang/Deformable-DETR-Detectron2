#!/bin/bash
set -xe

gpus=$1 #1
export DETECTRON2_DATASETS=YOUR_DATA_PATH

# Deformable DETR ResNet50
python tools/checkpoint_converter.py \
    --source_model assets/model_zoo/r50_deformable_detr-checkpoint.pth \
    --output_model assets/model_zoo/r50_deformable_detr-checkpoint_d2.pth
python train_net.py \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    --eval-only \
    --config-file config/deformable_detr_r50.yaml \
    --num-gpus $gpus \
    MODEL.WEIGHTS assets/model_zoo/r50_deformable_detr-checkpoint_d2.pth \

# Deformable DETR ResNet50 single scale
python tools/checkpoint_converter.py \
    --source_model assets/model_zoo/r50_deformable_detr_single_scale-checkpoint.pth \
    --output_model assets/model_zoo/r50_deformable_detr_single_scale-checkpoint_d2.pth
python train_net.py \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    --eval-only \
    --config-file config/deformable_detr_r50_single_scale.yaml \
    --num-gpus $gpus \
    MODEL.WEIGHTS assets/model_zoo/r50_deformable_detr_single_scale-checkpoint_d2.pth \

# Deformable DETR ResNet single scale DC5
python tools/checkpoint_converter.py \
    --source_model assets/model_zoo/r50_deformable_detr_single_scale_dc5-checkpoint.pth \
    --output_model assets/model_zoo/r50_deformable_detr_single_scale_dc5-checkpoint_d2.pth
python train_net.py \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    --eval-only \
    --config-file config/deformable_detr_r50_single_scale_dc5.yaml \
    --num-gpus $gpus \
    MODEL.WEIGHTS assets/model_zoo/r50_deformable_detr_single_scale_dc5-checkpoint_d2.pth \

# Deformable DETR ResNet50 plus iterative bbox refinement
python tools/checkpoint_converter.py \
    --source_model assets/model_zoo/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
    --output_model assets/model_zoo/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_d2.pth
python train_net.py \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    --eval-only \
    --config-file config/deformable_detr_r50_plus_iterative_bbox_refinement.yaml \
    --num-gpus $gpus \
    MODEL.WEIGHTS assets/model_zoo/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_d2.pth \

# Deformable DETR ResNet50 plus iterative bbox refinement plus two stage
# Warning: produces worse results than reported by the original paper: AP,AP50,AP75,APs,APm,APl=42.2739,59.7001,45.3141,27.4975,43.8332,58.0246
python tools/checkpoint_converter.py \
    --source_model assets/model_zoo/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth \
    --output_model assets/model_zoo/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint_d2.pth
python train_net.py \
    --dist-url tcp://127.0.0.1:$(( RANDOM % 1000 + 50000 )) \
    --eval-only \
    --config-file config/deformable_detr_r50_plus_iterative_bbox_refinement_plus_two_stage.yaml \
    --num-gpus $gpus \
    MODEL.WEIGHTS assets/model_zoo/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint_d2.pth \