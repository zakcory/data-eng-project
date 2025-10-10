#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_DETERMINISTIC=1

conda activate cs236207

# python pipeline.py --model_name lstm --dataset_name IMDB --fine_tune
python pipeline.py --model_name  resnet18 --dataset_name cifar10 --fine_tune