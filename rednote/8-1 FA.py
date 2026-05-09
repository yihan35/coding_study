# Pytorch 2.0+ 内置的FlashAttention（面试可提，但不建议写这个）
out = F.scaled_dot_product_attention(
    query=q,
    key=k,
    value=v,
    attn_mask=attn_mask,
    dropout_p=0.0,
    is_causal=False  # 如果是因果遮蔽，设为 True
)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(FlashAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)  # 缩放因子

        # 线性层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # 1. 计算 Q, K, V 的线性映射
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 分头并转置: (B, L, E) -> (B, H, L, D_h), 以便进行 MatMul
        def split_heads(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(q)  # (B, H, L, D_h)
        K = split_heads(k)  # (B, H, L, D_h)
        V = split_heads(v)  # (B, H, L, D_h)

        # 初始化输出: 最终输出沿着 Q 维度累加 (B, H, L, D_h)
        out = torch.zeros_like(Q)
        block_size = 64  # 分块大小, 可调整进行优化

        # 3. 初始化 Softmax 聚合所需的全局缓存 (沿 Q 维度 L)
        device = x.device
        # L_i / O_i (Logits Max) 和 Z_i (Softmax Sum)
        max_scores = torch.full((batch_size, self.num_heads, seq_len, 1), float('-inf'), device=device)
        softmax_sum = torch.zeros(batch_size, self.num_heads, seq_len, 1, device=device)

        # --- FlashAttention (Softmax 聚合 阶段) ---
        # 循环 K/V 块 (i) 和 Q 块 (j)
        for i in range(0, seq_len, block_size):  # i 遍历 K/V 块
            K_block = K[:, :, i: i + block_size]
            V_block = V[:, :, i: i + block_size]

            for j in range(0, seq_len, block_size):  # j 遍历 Q 块
                Q_block = Q[:, :, j: j + block_size]

                # 当前 Q 块对应的索引范围
                q_slice = slice(j, j + block_size)

                # 计算 Q_block 和 K_block 的相似度, 并进行缩放
                # attention_chunk 形状: (B, H, B_s, B_s)
                attention_chunk = torch.matmul(Q_block, K_block.transpose(-2, -1)) * self.scale

                # 4. Softmax 增量更新逻辑 (指数校正)
                # 计算当前块的局部最大值 M_ij
                M_ij = attention_chunk.max(dim=-1, keepdim=True)[0]
                # 获取 Q 块对应的全局最大值 L_j (旧最大值)
                L_j_old = max_scores[:, :, q_slice]
                # 计算新的全局最大值 L_j_new = max(L_j_old, M_ij)
                L_j_new = torch.maximum(L_j_old, M_ij)

                # 5. 指数校正和累积
                # 校正因子 exp(L_j_old - L_j_new)
                exp_correction_factor = torch.exp(L_j_old - L_j_new)

                # 更新 softmax_sum (Z_j_new = Z_j_old * exp(L_j_old - L_j_new) + sum(exp(A_ij - L_j_new))
                exp_scores = torch.exp(attention_chunk - L_j_new)
                softmax_sum_chunk = exp_scores.sum(dim=-1, keepdim=True)

                softmax_sum_old = softmax_sum[:, :, q_slice]
                softmax_sum[:, :, q_slice] = (softmax_sum_old * exp_correction_factor) + softmax_sum_chunk

                # 更新全局最大值
                max_scores[:, :, q_slice] = L_j_new

                # 6. 累积输出 (O_j = O_j * exp(L_j_old - L_j_new) + MatMul(exp(A_ij - L_j_new), V_block))
                # 校正旧的输出 (out[:, :, q_slice] 是 O_j_old)
                out[:, :, q_slice] *= exp_correction_factor

                # 计算新的输出块, 并累加
                out[:, :, q_slice] += torch.matmul(exp_scores, V_block)

        # 7. 最终归一化和线性投影
        # 最终归一化 O_j = O_j / Z_j
        out = out / softmax_sum

        # 通过线性层返回最终输出, 合并多头: (B, H, L, D_h) -> (B, L, E)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


# 测试代码
if __name__ == '__main__':
    x = torch.randn(2, 128, 512)  # 确保 seq_len 是 block_size 的倍数
    flash_attn = FlashAttention(embed_dim=512, num_heads=8)

    output = flash_attn(x)
    print(f"输出形状: {output.shape}")  # [2, 128, 512]