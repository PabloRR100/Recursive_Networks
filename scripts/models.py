
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
        self.act = nn.ReLU(inplace=True)
    
        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           
        
        self.W = nn.ModuleList(                                 
            [nn.Conv2d(32,32,3, padding=1) for _ in range(self.L)])   
        
        self.fc = nn.Linear(8*8*self.M, 10)
        
#        # Custom Initialization
##        named_parameters = net.named_parameters()
##        net = Conv_Net('test', layers=16, filters=32)
#
#        for name, param in self.named_parameters():
#            
#            # Vm has zero mean and 0.1 std (0.01 var)
#            if 'V' in name and 'weight' in name:
#                param.data.normal_(0, 0.1)
#            
#            # W are initialized with the identity matrix - Kronecker delta
#            elif 'W' in name and 'weight' in name:
#                param.data.fill_(0)
#                for i in range(32):
#                    param.data[i][0][0][0].fill_(1)
#                
#            ## TODO: C is not specified in the paper
#            elif 'fc' in name and 'bias' in name:
#                param.data.fill_(0)
        
    def forward(self, x):
        
        x = self.act(self.V(x))         # Out: 32x32xM  -- Maybe padding = 4?
        x = self.P(x)                   # Out: 8x8xM  -- Check also padding here
        for w in self.W:                
            x = self.act(w(x))          # Out: 8x8xM  -- Check also padding here)]
        x = x.view(x.size(0), -1)       # Out: 64*M  (M = 32 -> 2048)
        return self.fc(x)



class Conv_Recusive_Net(nn.Module):
    
    def __init__(self, name:str, layers:int, filters:int=32):
        super(Conv_Recusive_Net, self).__init__()
        
        self.name = name
        self.L = layers
        self.M = filters
        self.act = nn.ReLU()

        self.V = nn.Conv2d(3,self.M,8, stride=1, padding=3)     # Out: 32x32xM  -- Maybe padding = 4?
        self.P = nn.MaxPool2d(4, stride=4, padding=2)           # Out: 8x8xM  -- Check also padding here
        
        # WASAY: Shouldn't the kernel number here be M of size 3x3. Am I missing something here? 
        self.W = nn.Conv2d(32,32,3, padding=1)                 # Out: 8x8xM  -- Check also padding here)]
        
        self.fc = nn.Linear(8*8*self.M, 10)
        
    def forward(self, x):
        
        x = self.act(self.V(x))
        x = self.P(x)
        for w in range(self.L):
            x = self.act(self.w(x))
        # WASAY: "The final hidden layer is subject to pixel-wise L2 normalization", do we account for that?
        x = x.view(x.size(0), -1)
        return self.fc(x)       # WASAY: I was wondering why we don't apply softmax here before we retun the activations.
                                # This is clearly different from the simple convnet, where we do apply softmax.


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