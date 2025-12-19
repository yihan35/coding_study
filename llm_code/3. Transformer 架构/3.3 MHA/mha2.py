import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self,hidden_dim,num_heads):
        super(MultiheadAttention,self)._init_()
        assert hidden_dim % num_heads == 0 
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.k_linear = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.v_linear = nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.out = nn.Linear(hidden_dim,hidden_dim,bias=False)

        self.norm = math.sqrt(self.head_dim)
    
    def forward(self,x):
        bsz,seq_len,hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = q.reshape(bsz,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.reshape(bsz,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.reshape(bsz,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        att = torch.matmul(q,k.transpose(2,3))/self.norm
        att = torch.softmax(att,dim=-1)
        out = torch.matmul(att,v)
        out = out.transpose(1,2).contiguous().view(bsz,seq_len,self.hidden_dim)
        return self.out(out)
