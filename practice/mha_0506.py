import torch
import torch.nn as nn
import math
class MHA(nn.Module):
    def __init__(self,eps,head,d_model):
        self.eps = eps
        self.head = head
        self.d_model = d_model
        self.Q = nn.Linear(d_model,d_model)
        self.K = nn.Linear(d_model,d_model)
        self.V = nn.Linear(d_model,d_model)
    def forward(self,x):
        bsz,seq,d_model = x.shape
        head = self.head
        d = d_model // head
        xq = self.Q(x).reshape(bsz,seq,head,d).transpose(1,2)
        xk = self.K(x).reshape(bsz,seq,head,d).transpose(1,2)
        xv = self.V(x).reshape(bsz,seq,head,d).transpose(1,2)
        att = torch.matmul(xq,xk.tranpose(-1,-2))
        mask = torch.tril(torch.ones(seq,seq)).bool()
        att = att.masked_fill(~mask,float('-inf'))
        att_norm = torch.softmax(att/math.sqrt(d),dim =-1)
        out = torch.matmul(att_norm,xv).transpose(-1,-2).contiguous().view(bsz,seq,d_model)
        return self.O(out)