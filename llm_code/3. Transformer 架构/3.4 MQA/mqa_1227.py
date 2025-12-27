import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiQueryAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,self.d_k)
        self.v = nn.Linear(d_model,self.d_k)
        self.o = nn.Linear(d_model,d_model)
    def forward(self,x,mask=None):
        bsz,seq,d = x.shape
        q = self.q(x).reshape(bsz,seq,self.num_head,self.d_k).transpose(1,2)
        k = self.k(x).reshape(bsz,seq,1,self.d_k).transpose(1,2)
        v = self.v(x).reshape(bsz,seq,1,self.d_k).transpose(1,2)
        score = torch.matmul(q,k.transpose(2,3))/ math.sqrt(self.d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0,float('-inf'))
        score = F.softmax(score,dim = -1)
        output = torch.matmul(score,v)
        output = output.transpose(1,2).contiguous().view(bsz,seq,d)
        output = self.o(output)
        return output
bsz,seq,d = 2,4,8
num_head = 4
x = torch.randn(bsz,seq,d)
print("输入的形状：",x.shape)
mqa = MultiQueryAttention(d,num_head)
print("输出的形状：",mqa(x).shape)