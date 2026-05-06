import torch
import torch.nn.functional as F

# 统一参数
batch_size = 2
seq_len = 4
vocab_size = 10

# 模拟两个分布：学生模型 / 教师模型
logits_s = torch.randn(batch_size, vocab_size)
logits_t = torch.randn(batch_size, vocab_size)

# 实现 1：官方 API 实现
loss_kl_official = F.kl_div(
    F.log_softmax(logits_s, dim=-1),  # input: log_prob
    F.softmax(logits_t, dim=-1),      # target: prob
    reduction='batchmean'             # 数学上正确的 KL
)

# 实现 2：手动实现
def manual_kl_div(logits_s, logits_t):
    p = F.softmax(logits_t, dim=-1)  # 目标分布
    q = F.softmax(logits_s, dim=-1)  # 预测分布
    # KL(p||q) = sum(p * log(p/q))
    return torch.sum(p * torch.log(p / q), dim=-1).mean()

loss_kl_manual = manual_kl_div(logits_s, logits_t)