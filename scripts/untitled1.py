

from math import ceil, floor

def define_layers(Lr, Lo):
    
    # I need to split Lr layers in Lo buckets
    
    # If I have too many  
    if Lr - ceil(Lr/Lo)*(Lo-1) > 0:    
        r = ceil(Lr/Lo)
        res = Lr - r*(Lo-1)
        B = [r] * (Lo-1) + [res]
        
    # If I have too few    
    else:
        r = floor(Lr/Lo)
        res = Lr - r*(Lo-1)
        B = [r] * (Lo-1) + [res]

    return B


Lo = 6
Lr = 12
print(define_layers(Lr, Lo))
# Out: [2, 2, 2, 2, 2, 2]


Lo = 5
Lr = 12
print(define_layers(Lr, Lo))
# Out: [2, 2, 2, 2, 4]


Lo = 5
Lr = 13
print(define_layers(Lr, Lo))
# Out: [3, 3, 3, 3, 1]


Lo = 5
Lr = 32
print(define_layers(Lr, Lo))
# Out: [7, 7, 7, 7, 4]


Lo = 5
Lr = 33
print(define_layers(Lr, Lo))
# Out: [7, 7, 7, 7, 5]


Lr = 9
Lo = 5
print(define_layers(Lr, Lo))
# Out: [2, 2, 2, 2, 1]

Lr = 8
Lo = 5
print(define_layers(Lr, Lo))
# Out: [1, 1, 1, 1, 4]


