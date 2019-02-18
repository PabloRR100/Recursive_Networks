
import torch
from torch import nn

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

    
# Convolutional Networks
        
class Conv_Net(nn.Module):
    
    def __init__(self, name:str, layers:int, filters:int=32):
        super(Conv_Net, self).__init__()
        
        self.name = name
        self.L = layers
        self.M = filters
        self.act = nn.ReLU()
    
        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM  -- Maybe padding = 4?
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM  -- Check also padding here
        
        self.W = nn.ModuleList(                                # Out: 8x8xM  -- Check also padding here)]
            [nn.Conv2d(32,32,3, padding=1) for _ in range(self.L)])   
        
        self.fc = nn.Linear(8*8*self.M, 10)
        
        self.softmax = nn.Softmax()
        
        # Custom Initialization
        for name, param in self.named_parameters():
            
            # Vm has zero mean and 0.1 std (0.01 var)
            if 'V' in name and 'weight' in name:
                param.data.normal_().mul_(0.01)
            
            # W are initialized with the identity matrix - Kronecker delta
            elif 'W' in name and 'weight' in name:
                param.data.copy_(torch.eye(3))
                
            ## TODO: C is not specified in the paper
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)
        
        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for w in self.W:
            x = self.act(w(x))
        x = x.view(x.size(0), -1)
        return self.softmax(self.fc(x))



class Conv_Recusive_Net(nn.Module):
    
    def __init__(self, name:str, layers:int, filters:int=32):
        super(Conv_Recusive_Net, self).__init__()
        
        self.name = name
        self.L = layers
        self.M = filters
        self.act = nn.ReLU()
    
        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM  -- Maybe padding = 4?
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM  -- Check also padding here
        
        self.W = nn.Conv2d(32,32,3, padding=1)                 # Out: 8x8xM  -- Check also padding here)]
        
        self.fc = nn.Linear(8*8*self.M, 10)
        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for w in range(self.L):
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