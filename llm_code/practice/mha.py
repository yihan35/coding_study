import torch
import math
import torch.nn as nn
class MHA(nn.Module):# 类要继承 nn.Module
    def __init__(self,d_model,head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.Q = nn.Linear(d_model,d_model)
        self.K = nn.Linear(d_model,d_model)
        self.V = nn.Linear(d_model,d_model)
        self.O = nn.Linear(d_model,d_model)

    def forward(self,x,past_kv=None):
        bsz,seq,d_model = x.shape   # .shape 不是方法，不需要加括号调用
        head = self.head
        d = d_model // head
        xq = self.Q(x).reshape(bsz,seq,head,d).transpose(1,2)  # bsz,head,seq,d
        xk = self.K(x).reshape(bsz,seq,head,d).transpose(1,2)
        xv = self.V(x).reshape(bsz,seq,head,d).transpose(1,2)

        if past_kv is not None:
            past_k,past_v = past_kv
            xk = torch.cat([past_k],dim=2)
            xv = torch.cat([past_v],dim=2)
        present_kv = (xk,xv)
        total_seq = xk.shape[2]

        att = torch.matmul(xq,xk.transpose(-1,-2))  
        mask  = torch.tril(torch.ones(total_seq,total_seq)).bool() # 创建下三角矩阵
        # print(mask)

        # 对于decode阶段，只需要取最后 seq 行
        mask = mask[-seq:,:]

        att = att.masked_fill(mask==0,float('-inf')) 
        # print(att)
        att_norm =  torch.softmax(att / math.sqrt(d),dim =-1)  # 除以√d 后需要经过 softmax 再和 V 相乘
        out = torch.matmul(att_norm,xv).transpose(1,2).contiguous().view(bsz,seq,d_model)
        out_put = self.O(out)
        return out_put,present_kv

bsz,seq,d_model  =  2,4,8
head = 2
x = torch.randn(bsz,seq,d_model)
mha = MHA(d_model,head)
print(mha(x).shape)
print("kvcache:")



