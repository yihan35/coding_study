import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module): # 应该是 nn.Module
    def __init__(self, hidden_dim, num_heads=8): # 应该是 __init__
        super(MultiheadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden must be multiple of num head"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads # 每个头的维度
        
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self._norm_fact = math.sqrt(self.head_dim)

    def forward(self, x):
        # x 的维度: [bsz, seq_len, hidden_dim]
        bsz, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim
        
        # 1. 线性投射，得到 Q, K, V
        # 维度: [bsz, seq_len, hidden_dim]
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        
        # 2. 拆分成多头 (Multi-head)
        # 维度变换: [bsz, seq_len, hidden_dim] -> [bsz, seq_len, num_heads, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力分数 (Scaled Dot-Product Attention)
        # q 的维度: [bsz, num_heads, seq_len, head_dim]
        # k.transpose(2, 3) 的维度: [bsz, num_heads, head_dim, seq_len]
        # att 的维度: [bsz, num_heads, seq_len, seq_len]
        att = torch.matmul(q, k.transpose(2, 3)) / self._norm_fact
        att = torch.softmax(att, dim=-1)
        # att = F.dropout(att, p=0.5, training=self.training) # Dropout通常在注意力权重上使用
        
        # 4. 用注意力分数加权 V
        # att 的维度: [bsz, num_heads, seq_len, seq_len]
        # v 的维度: [bsz, num_heads, seq_len, head_dim]
        # out 的维度: [bsz, num_heads, seq_len, head_dim]
        out = torch.matmul(att, v)
        
        # 5. 拼接多头并重塑
        # 维度变换: [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, num_heads, head_dim] -> [bsz, seq_len, hidden_dim]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
        
        # 6. 最终的线性投射
        return self.out(out)