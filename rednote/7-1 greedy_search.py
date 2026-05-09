import torch
import torch.nn.functional as F


def greedy_search(model_logits, max_len=20, eos_token_id=2):
    """
    执行贪心搜索解码。

    Args:
        model_logits (torch.Tensor): 模型对一个序列预先计算好的logits,
                                    shape (batch_size, seq_len, vocab_size)。
                                    在实际生成中, 这一步是逐个token生成的。
        max_len (int): 最大生成长度。
        eos_token_id (int): 结束符的ID。

    Returns:
        torch.Tensor: 生成的词元序列。
    """
    batch_size = model_logits.size(0)
    # 存储生成的词元索引
    generated_sequence = torch.zeros(batch_size, max_len, dtype=torch.long)

    # 模拟逐个token生成的过程
    for t in range(max_len):
        # 获取当前时间步的 Logits
        current_logits = model_logits[:, t, :]

        # 计算概率分布(通过softmax)
        probs = F.softmax(current_logits, dim=-1)

        # 选择概率最高的词元索引
        next_token = torch.argmax(probs, dim=-1)

        # 将选择的词加入到生成的序列中
        generated_sequence[:, t] = next_token

        # 检查是否所有批次都遇到结束标记, 若遇到则提前停止
        if (next_token == eos_token_id).all():
            break

    return generated_sequence