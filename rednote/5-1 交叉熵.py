import torch
import torch.nn.functional as F

# 统一参数
batch_size = 2
seq_len = 4
vocab_size = 10

# 模型输出：未经过 softmax 的原始分数 (B, S, V)
logits = torch.randn(batch_size, seq_len, vocab_size)
# 真实标签：token id (B, S)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

# 实现 1：官方 API 
# 展平成 (B*S, V) 和 (B*S)
loss_ce_official = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1)
)

# 实现 2：手动实现
def manual_cross_entropy(logits, labels):
    # 1. 对词表维度做 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    # 2. 通过行和列的方式从 log_probs 二维矩阵中取出 token 对应的 log_prob
    # range(len(labels)) 表示行索引，labels 表示列索引 具体的 token id]
    selected_log_probs = log_probs[range(len(labels)), labels]
    # 3. 交叉熵 = -均值
    return -selected_log_probs.mean()

# 使用
loss_ce_manual = manual_cross_entropy(
    logits.reshape(-1, vocab_size),
    labels.reshape(-1)
)

# F.log_softmax 的手动实现
def stable_log_softmax(x, dim=-1):
    logsumexp_x = torch.logsumexp(x, dim=dim, keepdim=True)
    return x - logsumexp_x