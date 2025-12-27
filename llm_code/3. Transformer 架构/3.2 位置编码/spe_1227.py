import torch
import torch.nn as nn
import math
class SPE(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len,d_model)
        # shape [max_len,1]
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        print(position.shape)
        # shape [d_model/2]
        div = torch.exp(torch.arange(0,d_model,2,dtype=torch.float)*(-1/d_model)*math.log(10000))
        print(div.shape)
        pe[:,0::2] = torch.sin(position*div)
        pe[:,1::2] = torch.cos(position*div)
        # shape [0,len,d]
        print('pe 之前的形状：',pe.shape)
        pe = pe.unsqueeze(0)
        print('pe 之后的形状：',pe.shape)
        self.register_buffer('pe',pe)
    def forward(self,x):
        bsz,seq_len,d = x.shape
        x=x+self.pe[:,:seq_len:]
        return x

bsz=1
seq=2
d=8
x= torch.randn(bsz,seq,d)
print(x.shape)    
spe = SPE(d,seq)
spe(x)
print(x.shape)


