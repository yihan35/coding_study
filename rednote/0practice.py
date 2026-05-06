import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self,d_model,head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.o = nn.Linear(d_model,d_model)
    def forward(self,x):
        bsz,seq,d_model = x.shape
        d = d_model // self.head
        # [bsz,head,seq,d]
        wq = self.q(x).reshape(bsz,seq,self.head,d).transpose(1,2)
        wk = self.k(x).reshape(bsz,seq,self.head,d).transpose(1,2)
        wv = self.v(x).reshape(bsz,seq,self.head,d).transpose(1,2)
        # 计算相似度 [bsz,head,seq,seq]
        output1 = torch.matmul(wq,wk.transpose(-1,-2))/math.sqrt(d)
        mask = torch.tril(torch.ones(seq,seq)).bool()
        output1 = torch.softmax(output1.masked_fill(~mask,float('-inf')),dim=-1)
        # 计算注意力 [bsz,head,seq,d]
        output2 = torch.matmul(output1,wv)
        output3 = self.o(output2.transpose(1,2).contiguous().view(bsz,seq,d_model))
        return output3
bsz,seq,d_model = 2,4,8
x = torch.randn(bsz,seq,d_model)
mha = MHA(d_model,head = 2)
print(mha(x).shape)


class LayerNorm(nn.Module):
    def __init__(self,d,eps):
        super().__init__()
        self.eps = eps
        self.d = d
        self.a = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
    def forward(self,x):
        bsz,seq,d = x.shape
        # x - mean/ std+eps
        x_mean = x.mean(dim=-1,keepdim = True)
        x_var = x.mean(dim=-1,keepdim = True)
        x = (x-x_mean) / (x_std+self.eps)