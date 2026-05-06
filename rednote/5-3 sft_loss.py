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

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits:  (B, S, V) 模型输出
    labels:  (B, S)   真实 token 序列
    """
    # 核心：错位对应
    # logits[:, :-1, :]  预测  labels[:, 1:]
    shift_logits = logits[..., :-1, :].contiguous()  # (B, S-1, V)
    shift_labels = labels[..., 1:].contiguous()      # (B, S-1)

    # 展平计算交叉熵
    loss_1 = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=ignore_index
    )
    loss = manual_cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
    )
    return loss

def manual_cross_entropy(logits, labels):
    # 1. 对词表维度做 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    # 2. 通过行和列的方式从 log_probs 二维矩阵中取出 token 对应的 log_prob
    # range(len(labels)) 表示行索引，labels 表示列索引 具体的 token id]
    selected_log_probs = log_probs[range(len(labels)), labels]
    # 3. 交叉熵 = -均值
    return -selected_log_probs.mean()

# 使用
loss_sft = sft_loss(logits, labels)