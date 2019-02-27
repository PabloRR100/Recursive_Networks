#!/usr/bin/env bash
timestamp() {
  date +"%T"
}

python3 convnet.py --lr 0.1 1>./logs/temp.log 
