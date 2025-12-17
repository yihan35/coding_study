import torch
from torch import nn
class LayerNorm(nn.Module):
    def __init__(self,normalized_shape,eps=1e-5,elementwisw_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwisw_affine = elementwisw_affine
        if elementwisw_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self,x):
        # keepdim = True，能保证维度相同都是 2 维，但形状不相同
        # 广播规则：从最后一维开始匹配，只要 “维度大小相同” 或 “其中一个是 1”，就能广播
        # 在本例子中，x 和 mean 的最后一维分别是：3 vs 1，→ 1 可以扩展成 3；因此第 19 行可以计算
        mean = x.mean(dim = -1,keepdim = True) # mean/var 是 (2,1) 的张量，下面采用广播机制运算
        var = x.var(dim = -1,keepdim = True,unbiased = False)
        x_norm = (x-mean)/torch.sqrt(var+self.eps)
        if self.elementwisw_affine:
            return self.gamma* x_norm + self.beta
        return x_norm

x = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
layernorm = LayerNorm(x.shape[-1]) # 创建对象，初始化类
print(layernorm(x).shape)