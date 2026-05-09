import torch
import torch.nn.functional as F


def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    """
    执行Top-P（Nucleus）采样。

    Args:
        logits (torch.Tensor): 模型 logits, shape (batch_size, vocab_size)。
        top_p (float): 累积概率阈值。
        temperature (float): 温度。

    Returns:
        torch.Tensor: 采样得到的词元ID。
    """
    # 1. 应用温度并softmax
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # 2. 排序并计算累积概率
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 3. 找到超出阈值 p 的词元, 并将其移除
    # 创建一个mask, 保留累积概率小于p的词元
    sorted_indices_to_remove = cumulative_probs > top_p
    # 我们要保留第一个超过p的词元, 所以将 mask 右移一位
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 创建一个值为-inf的屏蔽张量
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')

    # 4. 再次softmax并采样
    final_probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(final_probs, num_samples=1)

    return next_token