# lyh practice 1219
import torch
import torch.nn as nn
class RMS(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self,x):
        x_new = torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)*x
        return x_new * self.weight
    
bsz,seq,d = 2,1,8
x = torch.rand(bsz,seq,d)
rms = RMS(d)
print(rms(x).shape)