
## NOTE: We make an assumption here that the ensemble size is the 
# same size or smaller than the size of the single deep Network.
# 
# simple class to simulate the number of parameters in a recursive Network (untied)

class Net:
    def __init__(self,M,L):
        self.M = M
        self.L = L
    
    def V(self):
        return 8*8*3*self.M
    
    def W(self):
        return (3*3*self.M*self.M*self.L) + (self.M*(self.L+1))

    def F(self):
        return (64*self.M*10) + 10
    
    def total(self):
        return self.V() + self.W() + self.F()
        
    
# Get the value of M keeping L the same as the deep Network:
def getM(S,K):
    ensemble_network = Net(M = S.M, L = S.L)
    budget = S.total()/K
    if K == 1:
        return S.M
        
    # print("Budget: " + str(budget))
    for M in range(S.M):
        ensemble_network.M = M
        if ensemble_network.total() > budget:
            return M-1

# Get the value of M given an L different from the deep network:
def getM_L(S,K,L):
    ensemble_network = Net(M = 1, L = L)
    budget = S.total()/K

    for M in range(S.M):
        ensemble_network.M = M
        if ensemble_network.total() == budget:
            return M
        if ensemble_network.total() > budget:
            return M-1
    return -1

# Get the value of L keeping M the same as the deep network:   
def getL(S,K):
    ensemble_network = Net(M = S.M, L = S.L)
    budget = S.total()/K

    for L in range(S.L):
        ensemble_network.L = L
        if ensemble_network.total() > budget:
            return L-1
    return L  ## TODO: M=1 is allowing to have Le > L for k=4 and returns Non e --> Maybe the mistake is in M=1 for ensemble network? shoutl be S.M
 
# Get the value of L keeping given an M different from the deep network:
def getL_M(S,K,M):
    ensemble_network = Net(M = M , L = S.L)
    budget = S.total()/K
    
    for L in range(S.L):
        ensemble_network.L = L
        if ensemble_network.total() == budget:
            return L
        if ensemble_network.total() > budget:
            return L-1
    return -1


if __name__ == '__main__':
    
    K = 4
    S = Net(M = 32, L = 16)
    print('\n\Single')
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Budget: ', S.total())
    
    ## A1: HORIZONTAL DIVISION: Fix L, K --> Divide M into Me
    Le = S.L
    Me = getM(S, K=K)
    Ek = Net(M = Me, L=Le)
    print('\n\nPLAIN HORIZONTAL')
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())
        
    ## A21: CONDITIONED HORIZONTAL DIVISION: Fix K, Choose Le < L --> Divide M into Me
    Le = 4
    Me = getM_L(S = S, K = 4, L = Le)
    Ek = Net(M = Me, L = Le)
    print('\n\nCONDITIONED HORIZONTAL Le < L')
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())
    
    ## A21: CONDITIONED HORIZONTAL DIVISION: Fix K, Choose Le > L --> Divide M into Me
    Le = 20
    Me = getM_L(S = S, K = 4, L = Le)
    Ek = Net(M = Me, L = Le)
    print('\n\nCONDITIONED HORIZONTAL Le > L')
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())


    ## B1: VERTICAL DIVISION: Fix M, K --> Divide L into Le
    Me = S.M
    Le = getL(S, K = 4)
    Ek = Net(M = Me, L = Le)
    print('\n\nPLAIN VERTICAL')
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())
    
    ## B21: CONDITIONED VERICAL DIVISION: Fix K, Choose Me < M --> Divide L into Le
    Me = 16
    Le = getL_M(S = S, K = 4, M = Me)
    Ek = Net(M = Me, L = Le)
    print('\n\nCONDITIONED VERTICAL Me < M')
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())
    
    ## B22: CONDITIONED VERICAL DIVISION: Fix K, Choose Me > M --> Divide L into Le
    Me = 48
    Le = getL_M(S = S, K = 4, M = Me)
    Ek = Net(M = Me, L = Le)
    print('\n\nCONDITIONED VERTICAL Me > M')
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())


    ## CA: RECURSIVE: Fix Le = 1, Choose M --> Calculate Ensemble Size allowed
    Le = 1
    Me = S.M
    Ek = Net(M = Me, L = Le)
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())
    
    ## CA: RECURSIVE: Fix Le = 1, Choose K --> Calculate Me allowed to use
    K = 6
    Le = 1
    Ek = Net(M = Me, L = Le)
    print(' S: M = {},  L = {}'.format(S.M, S.L))
    print('Ek: Me = {}, Le = {}, K = {}'.format(Me, Le, K))
    print('|S| / |Ek| = ', S.total() / Ek.total())
    
    
    

# SEARCH FOR THE CORRECT ARCHITECTURES
    
import numpy as np  
import matplotlib.pyplot as pl

    
# Plotting code
## Heatmap
def hm(matrix,xlabel="",ylabel="",title=""):
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title(title)
    pl.pcolor(matrix)
    pl.colorbar()
    pl.show()

## lineplot multiple yarrays
def lp(xarray,yarrays,labels,xlabel="",ylabel="",title=""):
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title(title)
    for yarray,label in zip(yarrays,labels):
        pl.plot(xarray,yarray,'o-',label=label)
    pl.legend()
    pl.show()
    
    
#############################################################
## SET UP 

M_S = 32
L_S = 16
S = Net(M=M_S, L=L_S) 

L_ = [4, 8, 16, 32, 64]
M_ = [4, 8, 16, 32, 64]
K_ = [4, 8, 16, 32, 64]

M_range = 64
L_range = 64
K_range = 64


## Experiment 1 -- Given a single deep network, sweep all values of L and K, calculate M

KLM = np.zeros((L_range, L_range))
KLNumParam = np.zeros((L_range, L_range))
KLNumParamNorm = np.zeros((L_range, L_range))

for K in range(1,K_range):
    for L in range(1,K_range):
        
        # Given K,L, and the deep network, compute M
        KLM[K,L] = getM_L(S,K,L) 
        
        # Compute total number of parameters in the ensemble
        temp_net = Net(M = KLM[K,L], L = L)
        KLNumParam[K,L] = K*temp_net.total()
        
        # Exception when M cannot be computed
        if KLM[K,L] == -1:
            KLNumParam[K,L] = 0

# Normalize KLNumParam
KLNumParamNorm = KLNumParam/np.max(KLNumParam)


# Find potential candidates

heatmap = True
lineplot = True
if heatmap:       
    pl.figure()
    hm(KLM,xlabel="K",ylabel="L",title="M | L,K")
    pl.figure()
    hm(KLNumParam,xlabel="K",ylabel="L",title="Total Parameters")
    pl.figure()
    hm(KLNumParamNorm,xlabel="K",ylabel="L",title="Normalize Total Parameters")
if lineplot:
    pl.figure()
    yarrays = [KLM[2,:][1:], KLM[4,:][1:], KLM[8,:][1:], KLM[16,:][1:], KLM[32,:][1:] ]
    labels = ['K=2','K=4','K=8','K=16','K=32']
    lp(range(1,33),yarrays,labels,xlabel="L",ylabel="M")


 

    
    
