#!/bin/bash

# Activate Environment
source activate pytorch_py36

# Big Ensemble

echo Running L=6 M=64 K=4
python train_non_Recursive_Ensemble.py -L 6 -M 64 -K 4 -E 4 |  K_4_L_6_M_64.txt
 
echo Running L=32 M=21 K=8
python train_non_Recursive_Ensemble.py -L 32 -M 21 -K 8 -E 4 |  K_8_L_32_M_21.txt

echo Running L=32 M=14 K=16
python train_non_Recursive_Ensemble.py -L 32 -M 14 -K 16 -E 4 |  K_16_L_32_M_14.txt

echo Running L=2 M=64 K=8
python train_non_Recursive_Ensemble.py -L 2 -M 64 -K 8 -E 4 |  K_8_L_3_M_64.txt  # Good candidate to train recursive

exit


