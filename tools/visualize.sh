#!/bin/bash
set -xe

output=$1 # saved/visualization
input=$2 #saved/deformable_detr_r50/inference/coco_instances_results.json
dataset=$3 # coco_2017_val
conf_threshold=$4 # 0.5
num_to_visualize=$5 # 1000
show_class=$6 # 1

export DETECTRON2_DATASETS=YOUR_DATA_PATH
python tools/visualize_json_results.py \
    --input ${input} \
    --output ${output}/${dataset}_conf${conf_threshold}_class${show_class} \
    --dataset ${dataset} \
    --conf-threshold ${conf_threshold} \
    --num-to-visualize ${num_to_visualize} \
    --show-class ${show_class}