import torch
import torch.nn as nn
import math
class MHA(nn.Module):
    def __init__(self,d,head):
        super().__init__()
        self.d = d
        self.head = head
        self.d_head = d // head
        self.Q = nn.Linear(d,d)
        self.K = nn.Linear(d,d)
        self.V = nn.Linear(d,d)
        self.O = nn.Linear(d,d)
    def forward(self,x):
        bsz,seq,d = x.shape
        q = self.Q(x).reshape(bsz,seq,self.head,self.d_head).transpose(1,2)
        k = self.K(x).reshape(bsz,seq,self.head,self.d_head).transpose(1,2)
        v = self.V(x).reshape(bsz,seq,self.head,self.d_head).transpose(1,2)
        att = torch.matmul(q,k.transpose(-1,-2))
        mask = torch.tril(torch.ones(seq,seq,device = x.device)).bool()
        att = att.masked_fill(~mask,float('-inf'))
        att_norm = torch.softmax(att/math.sqrt(self.d_head),dim =-1)
        output = torch.matmul(att_norm,v).transpose(1,2).contiguous().view(bsz,seq,d)
        return self.O(output)
bsz,seq,d = 2,3,4
head = 2
x = torch.randn(bsz,seq,d)
print(x)
mha = MHA(d,head)
print(mha(x).shape)

        

