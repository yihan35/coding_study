import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self,x):
        # .mean(dim=-1, keepdim=True) → 最后一维（8）算均值，保留维度 → 形状(2,4,1)
        # + eps（标量）→ 形状还是(2,4,1)（标量自动广播）
        x_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1,keepdim =True) + self.eps)
        return x_norm * self.weight

bsz,seq_len,d_model = 2,4,8
x=torch.rand(bsz,seq_len,d_model)
rmsnorm=RMSNorm(d_model)
output=rmsnorm(x)
# 以下逗号分隔（print 自动处理）
print("input data shape:",x.shape)
print("output data shape:",output.shape)
