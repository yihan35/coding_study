import torch

def softmax(x, dim=-1):
    # e^x_i / sum(e^x_j)
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def safe_softmax(x, dim=-1):
    # 减去最大值防止数值溢出: e^(x_i - max) / sum(e^(x_j - max))
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# 验证
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("输入:", x)
print("softmax:     ", softmax(x))
print("safe_softmax:", safe_softmax(x))
print("torch校验:   ", torch.softmax(x, dim=-1))

# 测试数值稳定性：极大值输入
x_large = torch.tensor([1000.0, 1001.0, 1002.0])
print("\n极大值输入:", x_large)
print("softmax (nan?):", softmax(x_large))       # 会出现 nan
print("safe_softmax:  ", safe_softmax(x_large))  # 正常