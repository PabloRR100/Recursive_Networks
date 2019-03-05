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
from models import Conv_Net, Conv_Recusive_Net, Conv_Custom_Recusive_Net, Conv_K_Recusive_Net

L = 16
M = 32
F = 16
K = 2

convnet = Conv_Net('ConvNet', L, M)
r_convnet = Conv_Recusive_Net('Recursive_ConvNet', L, M)
r_convnet_k = Conv_K_Recusive_Net('Custom_Recursive_ConvNet', L, M, K)
r_convnet_c = Conv_Custom_Recusive_Net('Custom_Recursive_ConvNet', L, M, F)

print('\n\nRegular ConvNet')
print('Parameters: {}M'.format(count_parameters(convnet)/1e6))
if comments: print(convnet)


print('\n\nRecursive ConvNet')
print('Parameters: {}M'.format(count_parameters(r_convnet)/1e6))
if comments: print(r_convnet)


print('\n\nRecursive Custom K_ConvNet')
print('Parameters: {}M'.format(count_parameters(r_convnet_k)/1e6))
if comments: print(r_convnet_k)


print('\n\nRecursive Custom ConvNet')
print('Parameters: {}M'.format(count_parameters(r_convnet_c)/1e6))
if comments: print(r_convnet_c)


# Calculate ensemble

E = round((count_parameters(convnet)/count_parameters(r_convnet)))
print('\n\nEnsemble size = ', E)

E = round((count_parameters(convnet)/count_parameters(r_convnet_c)))
print('\n\nCustom Ensemble size = ', E)


P1 = (8*8*3*M + 3*3*M**2*L + M*(L+1) + 64*M*10+10) * 1e-6
P2 = (8*8*3*M + 3*3*M**2   +   2*M   + 64*M*10+10) * 1e-6
P3 = (8*8*3*M + 3*3*M**2   + 2*M + 9*F**2 + 2*F + 64*F*10+10) * 1e-6

P = [P1, P2, P3]
[print('P{}: {}'.format(i,p)) for i,p in enumerate(P)]

'''
Future Work
-----------

Think of 'more intelligent' set ups in terms of:
    
    - Weights sharing between some layers of the ensemble
    - Apply recursivity from futher than the first layer 

'''


import pickle
import numpy as np
import matplotlib.pyplot as plt

#x = np.arange(0,128,32)
#y = 9*x**2 + 642*x
#
#plt.figure()
#plt.plot(x,y, c='red', label='n_params(F)')
#plt.axhline(640*M, c='blue', label='Normal recursive with M=32')
#plt.axhline(640*64, c='black', label='Normal recursive with M=64')
#plt.xlabel('F')
#plt.ylabel('# of Parameters')
#plt.legend()
#plt.show()



path1 = '../results/single_model/definitives/Results_Single.pkl'
path2 = '../results/single_recursive_model/Results_Single_Recursive.pkl'
path3 = '../results/ensemble_recursive_model/Results_Ensemble_Recursive.pkl'

with open(path1, 'rb') as input: results1 = pickle.load(input)
with open(path2, 'rb') as input: results2 = pickle.load(input)
with open(path3, 'rb') as input: results3 = pickle.load(input)


plt.figure()
plt.plot(range(len(results1.train_loss)), results1.train_loss, label='Train Untied')
plt.plot(range(len(results1.valid_loss)), results1.valid_loss, label='Valid Untied')
plt.plot(range(len(results2.train_loss)), results2.train_loss, label='Train Recursive')
plt.plot(range(len(results2.valid_loss)), results2.valid_loss, label='Valid Recursive')
plt.plot(range(len(results3.train_loss)), results3.train_loss, label='Train Recursive')
plt.plot(range(len(results3.valid_loss)), results3.valid_loss, label='Valid Recursive')
plt.title('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(results1.train_accy)), results1.train_accy, label='Train Untied')
plt.plot(range(len(results1.valid_accy)), results1.valid_accy, label='Valid Untied')
plt.plot(range(len(results2.train_accy)), results2.train_accy, label='Train Recursive')
plt.plot(range(len(results2.valid_accy)), results2.valid_accy, label='Valid Recursive')
plt.plot(range(len(results3.train_accy)), results3.train_accy, label='Train Recursive')
plt.plot(range(len(results3.valid_accy)), results3.valid_accy, label='Valid Recursive')
plt.title('Accuracy')
plt.legend()
plt.show()












