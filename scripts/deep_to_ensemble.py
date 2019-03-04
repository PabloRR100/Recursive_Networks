## NOTE: We make an assumption here that the ensemble size is the 
# same size or smaller than the size of the single deep network.
# 
# simple class to simulate the number of parameters in a recursive network (untied)
class net:
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
        
# Get the value of M keeping L the same as the deep network:
def getM(S,K):
    ensemble_network = net(M = 1, L = S.L)
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
    ensemble_network = net(M = 1, L = L)
    budget = S.total()/K

    # sweep M
    for M in range(S.M):
        ensemble_network.M = M
        if ensemble_network.total() == budget:
            return M
        if ensemble_network.total() > budget:
            return M-1
    return -1

# Get the value of L keeping M the same as the deep network:
def getL(S,K):
    ensemble_network = net(M = 1, L = S.L)
    budget = S.total()/K
    print("Budget: " + str(budget))
    for L in range(S.L):
        ensemble_network.L = L
        if ensemble_network.total() > budget:
            return L-1

# Get the value of L keeping given an M different from the deep network:
def getL_M(S,K,M):
    ensemble_network = net(M = M , L = S.L)
    budget = S.total()/K
    
    for L in range(S.L):
        ensemble_network.L = L
        if ensemble_network.total() == budget:
            return L
        if ensemble_network.total() > budget:
            return L-1
    return -1

# S = net(M = 32, L = 16)
# M_e = getM_L(S = S, K = 4, L = 4)
# e = net(M = M_e,L = S.L)

# print(e.M)
# print(e.L)
# print(e.total())