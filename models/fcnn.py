
import sys
import numpy as np
from sklearn.utils import shuffle
from torch import nn

''' 

Dimensionality
==============

# Inputs/Outputs are column vectors
X = (BS, inp) = (2, 1)
y = (out, 1)
y_one_hot = y* = (out, 1)

W1 (inp, hid) - (2,100) 
W2 (hid, out) - (100,2)

h = f(W1.T * X) = (hid, inp) * (inp, 1) = (hid, 1)
o = W2.T * h    = (out, hid) * (hid, 1) = (out, 1) 
p = softmax(o)  = (out,1)

L = SUM(y*log(p)) = (1)

dL/do = p - y* = (out, 1)
dL/dW2 = (do/dW2) * (dL/do).T = h * (p - y*).T = (hid, 1) * (1, out) = (hid, out)

dL/dh  = dL/do * dO/dh = (out, 1) = W2*(p - y*) = (hid, out) * (out, 1) = (hid, 1)
dL/dW1 = dL/dh * dh/dW = X * (W2(p - y*)).T = (inp, 1) * (1, hid) = (inp, hid)

W1 = W1 + lr * dW1
W2 = W2 + lr * dW2


Tracking the Network
====================

Update ratio of weights
-----------------------
A rough heuristic is that this ratio should be somewhere around 1e-3. 
If it is lower than this then the learning rate might be too low. 
If it is higher then the learning rate is likely too high.


Assumptions
===========

- All the layers are the same width
- All the layes share the same activation function

'''


class FC_Net(nn.Module):
    ''' Fully Connected Network '''
    
    def __init__(self, name:str, inp:int, out:int, hid:int,  n_layers:int):
        super(FC_Net, self).__init__()
                
        self.name = name
        self.lay_size = hid
        self.act = nn.ReLU()
        self.n_lay = n_layers
        
        self.fcI = nn.Linear(inp, hid, bias=False)        
        self.fcH = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(self.n_lay)])        
        self.fcO = nn.Linear(hid, out, bias=False)
                
    def forward(self, x):
                
        x = self.act(self.fcI(x))        
        for l in range(self.n_lay): 
            x = self.actf(self.fcHid[l](x))            
        return self.fcO(x)

    

##############################################################################
##############################################################################
        


class FC_Recursive_Net():
    ''' Fully Connected Network with Recursivity '''
    
        self.name = name
        self.lay_size = hid
        self.act = nn.ReLU()
        self.n_lay = n_layers
        
        self.fcI = nn.Linear(inp, hid, bias=False)        
        self.fcH = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(self.n_lay)])        
        self.fcO = nn.Linear(hid, out, bias=False)
                
    def forward(self, x):
                
        x = self.act(self.fcI(x))        
        for l in range(self.n_lay): 
            x = self.actf(self.fcHid[l](x))
            
#            # Recursive Layer (last layer)
#            if l == max(range(self.n_lay)) and self.recursive is not None:
#                for _ in range(self.recursive):
#                    x = self.actf(self.fcHid[l](x))
        return self.fcO(x)
