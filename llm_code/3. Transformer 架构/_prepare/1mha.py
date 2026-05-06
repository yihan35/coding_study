import torch
import math
import torch.nn as nn
class MHA(nn.Module):
    def __init__(self,dim,head):
        super().__init__()
        self.dim = dim
        self.head = head
        self.d = dim // head
        self.Q = nn.Linear(dim,dim)
        self.K = nn.Linear(dim,dim)
        self.V = nn.Linear(dim,dim)
        self.O = nn.Linear(dim,dim)
    
    def forward(self,x):
        #x:shape [bsz,seq,dim]
        bsz,seq,dim = x.shape
        # q k v 向量 并交换维度 [bsz,head,seq,d]
        q = self.Q(x).reshape(bsz,seq,self.head,self.d).transpose(1,2)
        k = self.K(x).reshape(bsz,seq,self.head,self.d).transpose(1,2)
        v = self.V(x).reshape(bsz,seq,self.head,self.d).transpose(1,2)
        # 计算注意力权重 [bsz,head,seq,seq]
        att = torch.matmul(q,k.transpose(-1,-2))
        x = att / math.sqrt(self.d) # 除以√d
        soft_x = torch.softmax(x,dim=-1)
        # 计算注意力得分 [bsz,head,seq,d] - > [bsz,seq,head,h] - > [[bsz,seq,dim]
        o = torch.matmul(soft_x,v).transpose(1,2).contiguous().view(bsz,seq,dim)
        # [bsz,seq,dim]
        output = self.O(o)
        return output
bsz,seq,dim = 2,4,10
head = 2
x = torch.randn(bsz,seq,dim)
print(x.shape)
mha = MHA(dim,head)
print(mha(x).shape)

        

