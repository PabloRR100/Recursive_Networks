#!/usr/bin/env bash
# 
python3 convnet.py --lr 0.1 -n lr_0_1 -e 300
python3 convnet.py --lr 0.01 -n lr_0_01 -e 300
python3 convnet.py --lr 0.001 -n lr_0_001 -e 300 
