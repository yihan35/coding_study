import torch

def beam_search(lm_prob, beam_size=3):
    """
    执行束搜索解码。

    Args:
        lm_prob (torch.Tensor): 模型输出的概率张量, shape (batch, seq_len, vocab_size)。
        beam_size (int): 束宽。

    Returns:
        tuple: (最终选择的序列索引, 对应的对数概率)。
    """
    batch, seq_len, vocab_size = lm_prob.shape

    # 为避免下溢并将连乘转换为连加, 对概率取对数
    log_lm_prob = torch.log(lm_prob)

    # --- 初始化 (t=0) ---
    # 取第一个时间步概率最高的k个token作为初始beam
    # log_beam_prob: (batch, beam_size)
    # indices: (batch, beam_size)
    log_beam_prob, indices = log_lm_prob[:, 0, :].topk(beam_size, sorted=True)

    # 将indices扩展一维, 用于后续拼接
    # indices: (batch, beam_size, 1)
    indices = indices.unsqueeze(-1)

    # --- 逐时间步扩展 Beam (t > 0) ---
    for i in range(1, seq_len):
        # 1. 扩展所有候选
        # log_beam_prob: (batch, beam_size) -> (batch, beam_size, 1)
        # log_lm_prob: (batch, vocab_size) -> (batch, 1, vocab_size)
        # current_log_probs: (batch, beam_size, vocab_size)
        # 这一步计算了 (beam_size个旧序列) + (所有新token) 的组合概率
        current_log_probs = log_beam_prob.unsqueeze(-1) + log_lm_prob[:, i, :].unsqueeze(1)

        # 2. 展平并取全局 top-k
        # current_log_probs: (batch, beam_size * vocab_size)
        log_beam_prob, flat_index = current_log_probs.view(batch, -1).topk(beam_size, sorted=True)

        # 3. 从展平的索引反推来源和新token
        # beam_id: 新beam来源于之前的哪个旧beam
        beam_id = flat_index // vocab_size
        # index: 新beam对应的真实token id
        index = flat_index % vocab_size

        # 4. 重新构建序列
        new_indices = []
        for j in range(batch):
            # 根据beam_id找到对应的历史序列, 并拼接上新的token id
            new_seqs_for_batch = torch.cat([indices[j][beam_id[j]], index[j].unsqueeze(-1)], dim=-1)
            new_indices.append(new_seqs_for_batch.unsqueeze(0))
        indices = torch.cat(new_indices, dim=0)

    return indices, log_beam_prob