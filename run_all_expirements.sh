#!/bin/bash

conda activate cs236207

python pipeline.py --model_name lstm --dataset_name IMDB
python pipeline.py --model_name  resnet18 --dataset_name cifar10