import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,d_model):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.d_k = d_model // num_head
        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)
        self.W_O = nn.Linear(d_model,d_model)

    def forward(self, x, mask=None):
        bsz,seq,d = x.shape
        q = self.W_Q(x).reshape(bsz,seq,self.num_head,self.d_k).transpose(1,2)
        k = self.W_K(x).reshape(bsz,seq,self.num_head,self.d_k).transpose(1,2)
        v = self.W_V(x).reshape(bsz,seq,self.num_head,self.d_k).transpose(1,2)
        scores = torch.matmul(q,k.transpose(-1,-2))/ math.sqrt(self.d_k) # 默写 sqrt
        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf')) # 默写 masked_fill
        scores_soft = F.softmax(scores,dim = -1)
        output = torch.matmul(scores_soft,v)
        # output = output.transpose(1,2).reshape(bsz,seq,self.d_model)
        output = output.transpose(1,2).contiguous().view(bsz,seq,self.d_model)
        output = self.W_O(output)
        return output
    
bsz,seq,d = 2,4,8
head = 4
x = torch.randn(bsz,seq,d)
print("输入的形状是：", x.shape)
mha = MultiHeadAttention(head,d)
output = mha(x)
print("输出的形状是：",output.shape)


        