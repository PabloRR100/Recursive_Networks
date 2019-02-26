
import math
import torch
from torch import nn

    
# Convolutional Networks
        
class Conv_Net(nn.Module):
    
    def __init__(self, name:str, layers:int, filters:int=32, normalize:bool=False):
        super(Conv_Net, self).__init__()
        
        self.name = name
        self.L = layers
        self.M = filters
        self.act = nn.ReLU(inplace=True)
        self.normalize = normalize # Wasay: Added batch normalization flag
    
        
        self.V = nn.Conv2d(3, self.M, 8, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(self.M)     
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           
        
        self.W = nn.ModuleList(                                 
            [nn.Conv2d(self.M,self.M,3, padding=1) for _ in range(self.L)])
        
        self.bn2 = nn.BatchNorm2d(self.M)
        
        self.fc = nn.Linear(8*8*self.M, 10)
        
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
                    m.bias.data.zero_() ## Notes (1)
                    
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
                
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
        self.bn1.weight.data.fill_(1)
        self.bn2.weight.data.fill_(1)
                        
    def forward(self, x):
        
        if self.normalize:
            x = self.act(self.bn1(self.V(x)))
        else:
            x = self.act(self.V(x))     # Out: 32x32xM  
        x = self.P(x)                   # Out: 8x8xM  
        for w in self.W:
            if self.normalize:
                x = self.act(self.bn2((w(x))))
            else:
                x = self.act(w(x))      # Out: 8x8xM  
        x = x.view(x.size(0), -1)       # Out: 64*M  (M = 32 -> 2048)
        return self.fc(x)



class Conv_Recusive_Net(nn.Module):
    
    def __init__(self, name:str, layers:int, filters:int=32):
        super(Conv_Recusive_Net, self).__init__()
        
        self.name = name
        self.L = layers
        self.M = filters
        self.act = nn.ReLU(inplace=True)

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM  
        
        self.W = nn.Conv2d(self.M,self.M,3, padding=1)          # Out: 8x8xM 
        
        self.fc = nn.Linear(8*8*self.M, 10)
        
        # Custom Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                        
        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for w in range(self.L):
            x = self.act(self.W(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)    
    

class Conv_K_Recusive_Net(nn.Module):
    '''
    Recursive block of K layers
    '''
    def __init__(self, name:str, layers:int, filters:int=32, k:int=2):
        super(Conv_Recusive_Net, self).__init__()
        
        self.name = name
        self.K = k
        self.L = layers
        self.M = filters
        self.act = nn.ReLU(inplace=True)

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM 
        
        self.W = nn.Conv2d(self.M,self.M,3, padding=1)          # Out: 8x8xM  
        
        self.fc = nn.Linear(8*8*self.M, 10)
        
        # Custom Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                        
        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for block in range(self.L/self.K):      # num_blocks = num_layers / layers_per_block
            for w in range(self.k):             # for each layer in the block
                x = self.act(self.W(x))
        x = x.view(x.size(0), -1)
        return self.fc(x) 



if '__name__' == '__main__':
    
    from torch.autograd import Variable
    def test(net):
        y = net(Variable(torch.randn(1,3,32,32)))
        print(y.size())
    
    L = 16
    M = 32
    
    convnet = Conv_Net('ConvNet', layers=L, filters=M)
    r_convnet = Conv_Recusive_Net('RecursiveConvNet', layers=L, filters=M)
    
    test(convnet)
    test(r_convnet)
    
    
    
# Fully Connected Networks

class FC_Net(nn.Module):
    ''' Fully Connected Network '''
    
    def __init__(self, name:str, inp:int, out:int, hid:int):
        super(FC_Net, self).__init__()
                
        self.name = name
        self.lay_size = hid
        self.act = nn.ReLU()
        
        self.fcI = nn.Linear(inp, hid, bias=True)        
        self.fcH = nn.Linear(hid, hid, bias=True)   
        self.fcO = nn.Linear(hid, out, bias=True)
                
    def forward(self, x):
                
        x = self.act(self.fcI(x))        
        x = self.act(self.fcH(x))            
        return self.fcO(x)
    

class FC_Recursive_Net(nn.Module):
    ''' Fully Connected Network with Recursivity '''
    
    def __init__(self, name:str, inp:int, out:int, hid:int, rec:int):
        super(FC_Recursive_Net, self).__init__()
    
        self.name = name
        self.n_lay = rec        
        self.act = nn.ReLU()
        assert rec > 0, 'Recursive parameters must be >= 1'
        
        self.fcI = nn.Linear(inp, hid, bias=True)        
        self.fcH = nn.Linear(hid, hid, bias=True) 
        self.fcO = nn.Linear(hid, out, bias=True)
                
    def forward(self, x):
                
        x = self.act(self.fcI(x))        
        x = self.act(self.fcH(x))
        for l in range(self.n_lay): 
            x = self.actf(self.fcHid(x))            
        return self.fcO(x)
