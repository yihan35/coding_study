import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 修正类名拼写
class MultiQueryAttention(nn.Module):
    """
    多查询注意力（MQA）模块。
    所有查询头（Query heads）共享同一套键（Key）和值（Value）头，以节省内存和计算。
    """
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiQueryAttention, self).__init__()
        # 维度必须能被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim must be multiple of num_heads"
        
        # --- 属性定义 ---
        self.hidden_dim = hidden_dim  # 修正拼写: hideen_dim -> hidden_dim
        self.num_heads = num_heads      # Q头的数量
        self.head_dim = hidden_dim // num_heads  # 每个头的维度
        
        # --- 核心线性层定义 ---
        # 1. Q的投影层：保持不变，投影到完整维度，后续拆分成 num_heads 个头
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 2. K的投影层：【MQA的核心】只投影到单个头的维度，因为所有Q头要共享它
        self.linear_k = nn.Linear(hidden_dim, self.head_dim, bias=False) # 修正：输出维度是 head_dim
        
        # 3. V的投影层：【MQA的核心】同样只投影到单个头的维度
        self.linear_v = nn.Linear(hidden_dim, self.head_dim, bias=False) # 修正：输出维度是 head_dim
        
        # 4. 最终输出层
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 注意力分数的缩放因子
        self._norm_fact = math.sqrt(self.head_dim)

    # 修正语法：forward方法需要与__init__同级
    def forward(self, x):
        # x 的输入维度: [bsz, seq_len, hidden_dim]
        bsz, seq_len, _ = x.shape
 
        # --- 1. 获取 Q, K, V ---
        # Q: 保持多头。维度变换: [bsz, seq_len, hidden_dim] -> [bsz, seq_len, num_heads, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        q = self.linear_q(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K: 只有一个头。维度变换: [bsz, seq_len, head_dim] -> [bsz, seq_len, 1, head_dim] -> [bsz, 1, seq_len, head_dim]
        k = self.linear_k(x).reshape(bsz, seq_len, 1, self.head_dim).transpose(1, 2) # 修正拼写: tranpose -> transpose
        
        # V: 也只有一个头。维度变换: [bsz, seq_len, head_dim] -> [bsz, seq_len, 1, head_dim] -> [bsz, 1, seq_len, head_dim]
        v = self.linear_v(x).reshape(bsz, seq_len, 1, self.head_dim).transpose(1, 2)
 
        # --- 2. 计算注意力 ---
        # 通过广播机制，让 num_heads 个Q头 与 1个K头 计算注意力分数
        # 维度: (bsz, num_heads, seq_len, head_dim) @ (bsz, 1, head_dim, seq_len) -> (bsz, num_heads, seq_len, seq_len)
        att = torch.matmul(q, k.transpose(2, 3))
        
        # 缩放，防止梯度消失
        att = att / self._norm_fact
        
        # Softmax，得到归一化的注意力权重
        att = torch.softmax(att, dim=-1)
        
        # (可选) Dropout，防止过拟合
        # att = F.dropout(att, p=0.1, training=self.training)
 
        # --- 3. 加权求和 ---
        # 用注意力权重乘以V。同样利用广播机制，让 num_heads 组权重与 1个V 进行计算
        # 维度: (bsz, num_heads, seq_len, seq_len) @ (bsz, 1, seq_len, head_dim) -> (bsz, num_heads, seq_len, head_dim)
        out = torch.matmul(att, v)
        
        # --- 4. 拼接并输出 ---
        # 将多头的结果拼接起来，恢复成原始输入维度
        # 维度: [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, num_heads, head_dim] -> [bsz, seq_len, hidden_dim]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)
 
        return self.out(out)