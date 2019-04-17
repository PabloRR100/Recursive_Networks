#!/bin/bash

# Activate Environment
source activate pytorch_p36
# source activate pytorch

E=700
BN=False

function g() {
    arg1=$1; arg2=$2; arg3=$3 
    git add .
    git commit -m 'changes after training K_${arg1}_L_${arg2}_M_${arg3}'
    git push
}

L=12; M=48; K=4;
echo Running L=$L M=$M K=$K
echo Y | python train_non_Recursive_Ensemble.py -L $L -M $M -K $K -E $E | tee K_${K}_L_${L}_M_${M}.txt
# g "$K" "$L" "$M"


L=5; M=48; K=8;
echo Running L=$L M=$M K=$K
echo Y | python train_non_Recursive_Ensemble.py -L $L -M $M -K $K -E $E | tee K_${K}_L_${L}_M_${M}.txt
# g "$K" "$L" "$M"


L=3; M=48; K=12;
echo Running L=$L M=$M K=$K
echo Y | python train_non_Recursive_Ensemble.py -L $L -M $M -K $K -E $E | tee K_${K}_L_${L}_M_${M}.txt
# g "$K" "$L" "$M"


L=1; M=48; K=16;
echo Running L=$L M=$M K=$K
echo Y | python train_non_Recursive_Ensemble.py -L $L -M $M -K $K -E $E | tee K_${K}_L_${L}_M_${M}.txt
# g "$K" "$L" "$M"


# Recursives

L=3; M=48; K=16;
echo Running L=$L M=$M K=$K
echo Y | python train_Recursive_Ensemble.py -L $L -M $M -K $K -E $E | tee K_${K}_L_${L}_M_${M}.txt
# g "$K" "$L" "$M"


L=5; M=48; K=16;
echo Running L=$L M=$M K=$K
echo Y | python train_Recursive_Ensemble.py -L $L -M $M -K $K -E $E | tee K_${K}_L_${L}_M_${M}.txt
# g "$K" "$L" "$M"


L=12; M=48; K=16;
echo Running L=$L M=$M K=$K
echo Y | python train_Recursive_Ensemble.py -L $L -M $M -K $K -E $E | tee K_${K}_L_${L}_M_${M}.txt
# g "$K" "$L" "$M"

exit


