import torch
import torch.nn as nn
class LayNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True)
        x_2 = (x-mean)/torch.sqrt(var+self.eps)
        x_out = x_2 * self.alpha + self.beta
        return x_out
    
bsz,seq,d = 2,2,4
laynorm = LayNorm(d)
x = torch.randn(bsz,seq,d)
print(laynorm(x).shape)