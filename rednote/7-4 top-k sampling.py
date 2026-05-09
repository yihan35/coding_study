import torch
import torch.nn.functional as F


def top_k_sampling(logits, top_k=50, temperature=1.0):
    """
    执行Top-K采样。

    Args:
        logits (torch.Tensor): 模型 logits, shape (batch_size, vocab_size)。
        top_k (int): 保留的候选项数量。
        temperature (float): 温度。

    Returns:
        torch.Tensor: 采样得到的词元ID。
    """
    # 1. 将概率非常低的词元的logits设为负无穷
    # 取出top_k个logits值
    top_k = min(top_k, logits.size(-1))  # 确保k不大于词汇表大小
    top_k_values, _ = torch.topk(logits, top_k)

    # 获取第k个值作为阈值
    kth_value = top_k_values[:, [-1]]

    # 将所有小于阈值的logits移除
    logits[logits < kth_value] = float('-inf')

    # 2. 应用温度并softmax
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # 3. 从新的分布中采样
    idx_next = torch.multinomial(probs, num_samples=1)

    return idx_next