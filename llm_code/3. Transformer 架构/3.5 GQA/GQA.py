import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (GQA) 模块。
    Q头被分成几组，每组内的Q头共享同一套K头和V头。
    """
    def __init__(self, hidden_dim, q_heads=8, kv_heads=2):
        super(GroupedQueryAttention, self).__init__()
        # Q头的数量必须是K/V头数量的整数倍
        assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads"
        
        # --- 属性定义 ---
        self.hidden_dim = hidden_dim
        self.q_heads = q_heads          # Q头的总数 (例如8)
        self.kv_heads = kv_heads        # K和V头的总数 (例如2)
        self.head_dim = hidden_dim // q_heads # 每个头的维度 (例如64)
        
        # 计算分组数量，即每个K/V头要被多少个Q头共享
        self.num_groups = q_heads // kv_heads # 例如 8 // 2 = 4

        # --- 核心线性层定义 ---
        # 1. Q的投影层：投影到所有Q头的总维度
        self.linear_q = nn.Linear(hidden_dim, self.q_heads * self.head_dim, bias=False)
        
        # 2. K的投影层：只投影到K/V头的总维度
        self.linear_k = nn.Linear(hidden_dim, self.kv_heads * self.head_dim, bias=False)
        
        # 3. V的投影层：同样只投影到K/V头的总维度
        self.linear_v = nn.Linear(hidden_dim, self.kv_heads * self.head_dim, bias=False)
        
        # 4. 最终输出层
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self._norm_fact = math.sqrt(self.head_dim)

    def forward(self, x):
        # x 的输入维度: [bsz, seq_len, hidden_dim]
        bsz, seq_len, _ = x.shape
 
        # --- 1. 获取 Q, K, V ---
        # Q: 维度变换: ... -> [bsz, seq_len, q_heads, head_dim] -> [bsz, q_heads, seq_len, head_dim]
        q = self.linear_q(x).reshape(bsz, seq_len, self.q_heads, self.head_dim).transpose(1, 2)
        
        # K: 维度变换: ... -> [bsz, seq_len, kv_heads, head_dim] -> [bsz, kv_heads, seq_len, head_dim]
        k = self.linear_k(x).reshape(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        
        # V: 维度变换: ... -> [bsz, seq_len, kv_heads, head_dim] -> [bsz, kv_heads, seq_len, head_dim]
        v = self.linear_v(x).reshape(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        
        # --- 2. 【GQA核心步骤】扩展K和V以匹配Q的分组 ---
        # k: [bsz, kv_heads, seq_len, head_dim] -> [bsz, q_heads, seq_len, head_dim]
        # v: [bsz, kv_heads, seq_len, head_dim] -> [bsz, q_heads, seq_len, head_dim]
        # repeat_interleave 会在指定维度上将每个元素重复 num_groups 次
        # 这就实现了让每组内的Q头都能“看到”同一个K/V头
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # --- 3. 计算注意力 (现在和MHA完全一样了) ---
        # 经过上一步的扩展，Q, K, V的头数量维度都对齐了 (都等于q_heads)
        # 维度: (bsz, q_heads, seq_len, head_dim) @ (bsz, q_heads, head_dim, seq_len) -> (bsz, q_heads, seq_len, seq_len)
        att = torch.matmul(q, k.transpose(2, 3))
        att = att / self._norm_fact
        att = torch.softmax(att, dim=-1)
 
        # --- 4. 加权求和 ---
        # 维度: (bsz, q_heads, seq_len, seq_len) @ (bsz, q_heads, seq_len, head_dim) -> (bsz, q_heads, seq_len, head_dim)
        out = torch.matmul(att, v)
        
        # --- 5. 拼接并输出 ---
        # 将多头的结果拼接起来，恢复成原始输入维度
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
 
        return self.out(out)