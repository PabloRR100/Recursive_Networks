
import math
import torch
from torch import nn

    
# Convolutional Networks
        
class Conv_Net(nn.Module):
    
    def __init__(self, name:str, L:int=16, M:int=64, normalize:bool=False):
        super(Conv_Net, self).__init__()
        
        self.L = L
        self.M = M
        self.name = name
        self.normalize = normalize
        
        self.act = nn.ReLU()    
        
        if normalize:
            self.d1 = nn.Dropout2d(p=0.2)
            self.d2 = nn.Dropout2d(p=0.5)
            self.bn1 = nn.BatchNorm2d(num_features=self.M)
            self.bn2 = nn.BatchNorm2d(num_features=self.M)
        
        self.V = nn.Conv2d(3, self.M, 8, stride=1, padding=3)
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           
        
        self.W = nn.ModuleList(                                 
            [nn.Conv2d(self.M,self.M,3, padding=1) for _ in range(self.L)])
                
        self.C = nn.Linear(8*8*self.M, 10)
        
        # NOT FOLLOWING PAPER Custom Initialization
        '''
        NOTES:
            [0] - This has been changed from : for param, name in ... -> Review if doesn't work
            [1] - Read on DL book that with RELU is preferable to start biases with 0.01
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0.01)
                
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.01)
               

    def forward(self, x):
            
        if not self.normalize:
            x = self.act(self.V(x))         # Out: 32x32xM
            print(x.shape)
            x = self.P(x)                   # Out: 8x8xM  
            print(x.shape)
            for w in self.W:
                x = self.act(w(x))          # Out: 8x8xM  
                print(x.shape)
            x = x.view(x.size(0), -1)       # Out: 64*M  (M = 32 -> 2048)
            print(x.shape)
            return self.C(x)
        
        else:
        
            x = self.bn1(self.act(self.V(x)))           # Out: 32x32xM  
            x = self.P(x)                               # Out: 8x8xM  
            for w in self.W:
                x = self.bn2(self.act(w(x)))            # Out: 8x8xM  
            x = x.view(x.size(0), -1)                   # Out: 64*M  (M = 32 -> 2048)
            return self.C(x)


class Conv_Recusive_Net(nn.Module):
    
    def __init__(self, name:str, L:int, M:int=32):
        super(Conv_Recusive_Net, self).__init__()
        
        self.L = L
        self.M = M
        self.name = name
        self.act = nn.ReLU(inplace=True)

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM  
        self.W = nn.Conv2d(self.M,self.M,3, padding=1)          # Out: 8x8xM 
        self.C = nn.Linear(8*8*self.M, 10)
        
        # Custom Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                        
    
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for w in range(self.L):
            x = self.act(self.W(x))
        x = x.view(x.size(0), -1)
        return self.C(x)  
   


class Conv_K_Recusive_Net(nn.Module):
    '''
    Recursive block of K layers
    '''
    def __init__(self, name:str, Lo:int, Lr:int, M:int=32):
        super(Conv_K_Recusive_Net, self).__init__()
        
        self.name = name

        self.M = M
        self.Lo = Lo
        self.Lr = Lr
        self.R = math.ceil(Lr/Lo) # Recursivity withing each block
        self.act = nn.ReLU(inplace=True)
        self.B = [self.R] * (Lo-1) + [Lr%self.R] if Lr%self.R != 0 else [self.R] * Lo

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM 
        
        self.Wk = nn.ModuleList(                                 
            [nn.Conv2d(self.M,self.M,3, padding=1) for _ in range(len(self.B))])  # Out: 8x8xM  
        
        self.C = nn.Linear(8*8*self.M, 10)
        
        # Custom Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                        
        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        
        for b,W in zip(self.B, self.Wk):    # for each layer
            for _ in range(b):              # run a recursive block
                x = self.act(W(x))
    
        x = x.view(x.size(0), -1)
        return self.C(x) 


if '__name__' == '__main__':
    
    from torch.autograd import Variable
    def test(net):
        y = net(Variable(torch.randn(1,3,32,32)))
        print(y.size())
    
    L = 16
    M = 32
    K = 2
    F = 16
    
    M = 4
    Lo = 5
    Lr = 30

    
    convnet = Conv_Net('ConvNet', L, M)
    r_convnet = Conv_Recusive_Net('RecursiveConvNet', L, M)
    r_convnet_k = Conv_K_Recusive_Net('Custom_Recursive_ConvNet', Lo, Lr, M)

    test(convnet)
    test(r_convnet)
    test(r_convnet_k)
    
    exit()
    
    
    
    
    
    class Ensemble:
        
        def __init__(self, net:list, size:int=None):
            super(Ensemble).__init__()
            
            if size is None:
                assert isinstance(net, list), \
                'Models should be a list if size is not provided'
                self.nets = net
                self.size = len(net)
                
            else:
                assert not isinstance(net, list), \
                'If size is provide, pass just a single Model'
                self.net = [net('n{}'.format(n) for n in range(size))]
                self.size = size
                
        def train(self):
            for net in self.nets:
                net.train()
                    
        def eval(self):
            for net in self.nets:
                net.eval()
                
        def forward(self, x, device):
            '''
            :Input: Tensor
            :Output: List of predictions for each model and the ensemble'
            '''
            outputs = list()
            for n, net in enumerate(self.net):
                outputs.append(net(x))
            return outputs
    
    
    net1 = Conv_Net('n1', 32, 64)
    net2 = Conv_Net('n2', 32, 64)
    net3 = Conv_Net('n3', 32, 64)
    nets = [net1, net2, net3]        
    
    ensemble1 = Ensemble(nets)
    ensemble2 = Ensemble(Conv_Net, 3)
    ensembles = [ensemble1, ensemble2]
    
    images, labels = next(iter(trainloader))