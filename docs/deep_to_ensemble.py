import math as math
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

# Analytical version of getM_L
def getM_L_analytical(S,K,L):
    S_ = S.total()
    #
    sqrt_terms = math.sqrt((L*((36*S_*1.0)/K*1.0) + L + 1306) + 693889)
    nominator = sqrt_terms - L - 833
    denominator = 18*L*1.0
    M = nominator/denominator
    if M > S.M:
        return -1
    return math.floor(M)

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

# Analytical version of getL_M 
def getL_M_analytical(S,K,M):
    S_ = S.total()
    #
    nominator = S_ - (833 * K * M) - (10 * K)
    denominator = K * M *( (9 * M) + 1)
    L = nominator*1.0/denominator*1.0
    return math.floor(L)






