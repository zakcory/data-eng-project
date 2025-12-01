#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_DETERMINISTIC=1

source .venv/bin/activate

python pipeline.py --model_name beannet --dataset_name drybean --fine_tune
python pipeline.py --model_name lstm --dataset_name IMDB --fine_tune
python pipeline.py --model_name  resnet18 --dataset_name cifar10 --fine_tune
