import torch

# 模拟数据
logits = torch.randn(3, 5)
labels = torch.tensor([2, 4, 0])

def softmax(logits):
    max_val = torch.max(logits, dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - max_val)
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

def cross_entropy(logits, labels):
    probs = softmax(logits)

    one_hot = torch.zeros_like(probs)
    rows = torch.arange(len(labels))
    one_hot[rows, labels] = 1.0 # 构建 one hot 矩阵

    log_probs = torch.log(probs)
    cross_entropy = -torch.sum(one_hot * log_probs) / len(probs)
    return cross_entropy

print(cross_entropy(logits,labels))
