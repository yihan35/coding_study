import torch
import torch.nn.functional as F


def temperature_sampling(logits, temperature=1.0):
    """
    执行温度采样。

    Args:
        logits (torch.Tensor): 模型在当前步的输出, shape (batch_size, vocab_size)。
        temperature (float): 温度值, 必须大于0。

    Returns:
        torch.Tensor: 采样得到的词元ID。
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    # 1. 对logits进行缩放
    scaled_logits = logits / temperature

    # 2. softmax得到概率分布
    probs = F.softmax(scaled_logits, dim=-1)

    # 3. 从调整后的分布中进行多项式采样
    sampled_token = torch.multinomial(probs, num_samples=1)

    return sampled_token