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
    ensemble_network = Net(M = 1, L = S.L)
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
    ensemble_network = Net(M = 1, L = S.L)
    budget = S.total()/K
    print("Budget: " + str(budget))
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
    
    S = Net(M = 32, L = 16)
    print(S.total())
    
    ## A: HORIZONTAL DIVISION: Fix L, K --> Divide M 
    Le = S.L
    Me = getM(S, K=4)
    Ek = Net(M = Me, L=Le)
    print(S.total() / Ek.total())
    
    ## B: VERTICAL DIVISION: Fix M, K --> Divide L
    Me = S.M
    Le = getL(S, K = 4)
    Ek = Net(M = Me, L = Le)
    print(S.total() / Ek.total())
    
    ## AB: CONDITIONED HORIZONTAL DIVISION: Fix K, Choose L --> Divide M
    Le = 4
    Me = getM_L(S = S, K = 4, L = Le)
    Ek = Net(M = Me, L = Le)
    print(S.total() / Ek.total())
    
    ## BA: CONDITIONED VERICAL DIVISION: Fix K, Choose M --> Divide L
    Me = 16
    Le = getL_M(S = S, K = 4, M = Me)
    Ek = Net(M = Me, L = Le)
    print(S.total() / Ek.total())

    ## CA: RECURSIVE: Fix Le = 1, Choose M --> Calculate Ensemble Size allowed
    Le = 1
    Me = S.M
    Ek = Net(M = Me, L = Le)
    print(S.total() / Ek.total())
    
    ## CA: RECURSIVE: Fix Le = 1, Choose K --> Calculate Me allowed to use
    K = 6
    Le = 1
    Ek = Net(M = Me, L = Le)
    print(S.total() / Ek.total())
    
    
    
    


