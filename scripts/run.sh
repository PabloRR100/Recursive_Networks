#!/bin/bash

# Activate Environment
# source activate pytorch_p36

# ensemble_prmts = {'L': 12,  'M': 48, 'BN': False, 'K': 4}       # - TODO
# ensemble_prmts = {'L': 5,   'M': 48, 'BN': False, 'K': 8}       # - TODO
# ensemble_prmts = {'L': 3,   'M': 48, 'BN': False, 'K': 12}      # - TODO
# ensemble_prmts = {'L': 1,   'M': 48, 'BN': False, 'K': 16}      # - TODO & RECURSIVE


# Big Ensemble
E=2
BN=False
L=12; M=48; K=4;


echo Running L=$L M=$M K=$K
echo Y | python train_non_Recursive_Ensemble.py -L $L -M $M -K $K -E $KE | tee K_$K_L_$L_M_$K.txt
 
#echo Running L=32 M=21 K=8
#echo Y | python train_non_Recursive_Ensemble.py -L 32 -M 21 -K 8 | tee K_8_L_32_M_21_IIII.txt

#echo Running L=32 M=14 K=16
#echo Y | python train_non_Recursive_Ensemble.py -L 32 -M 14 -K 16 | tee K_16_L_32_M_14_II.txt

#echo Running L=2 M=64 K=8
#echo Y | python train_non_Recursive_Ensemble.py -L 2 -M 64 -K 8 | tee K_8_L_3_M_64.txt  # Good candidate to train recursive

#echo Running L=2 M=64 K=8 Recursice
#echo Y | python train_Recursive_Ensemble.py -L 2 -M 64 -K 8 | tee K_8_L_3_M_64.txt  # Good candidate to train recursive

exit


