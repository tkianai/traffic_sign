#!/bin/bash

python models.py -a resnet50 --dist-url 'tcp://127.0.0.1:12345' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
--pretrained ../data/TrafficSign/images