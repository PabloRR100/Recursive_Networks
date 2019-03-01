#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This Script is used to create ensembles of recursive networks to match
the number of parameters of a single CNN given its:
    
    - L -> # of layers
    - M -> # of filters
    - E -> size of the ensemble

"""

import argparse
parser = argparse.ArgumentParser(description='Ensembles of Recursive Networks')
parser.add_argument('--filters', '-M', default=32, type=int, help='# of filters')
parser.add_argument('--layers', '-L', default=16, type=int, help='# of layers')
parser.add_argument('--comments', '-c', default=True, type=bool, help='print all the statements')


# Models 
# ------
    
# OPT 1: NO SHARING of any layers withing the ensemble
    
comments = True
from utils import count_parameters
from models import Conv_Net, Conv_Recusive_Net, Conv_Custom_Recusive_Net

L = 16
M = 32
F = 32

convnet = Conv_Net('ConvNet', layers=L, filters=M)
r_convnet = Conv_Recusive_Net('Recursive_ConvNet', L, M)
r_convnet_c = Conv_Custom_Recusive_Net('Custom_Recursive_ConvNet', L, M, F)

print('\n\nRegular ConvNet')
print('Parameters: {}M'.format(count_parameters(convnet)/1e6))
if comments: print(convnet)

print('\n\nRecursive ConvNet')
print('Parameters: {}M'.format(count_parameters(r_convnet)/1e6))
if comments: print(r_convnet)

print('\n\nRecursive Custom ConvNet')
print('Parameters: {}M'.format(count_parameters(r_convnet_c)/1e6))
if comments: print(r_convnet_c)


E = round((count_parameters(convnet)/count_parameters(r_convnet)))
print('\n\nEnsemble size = ', E)

P1 = (8*8*3*M + 3*3*M**2*L + M*(L+1) + 64*M*10+10) * 1e-6
P2 = 16 * ((8*8*3*M + 3*3*M**2 + M*2 + 64*M*10+10) * 1e-6)


'''
Future Work
-----------

Think of 'more intelligent' set ups in terms of:
    
    - Weights sharing between some layers of the ensemble
    - Apply recursivity from futher than the first layer 

'''


exit()