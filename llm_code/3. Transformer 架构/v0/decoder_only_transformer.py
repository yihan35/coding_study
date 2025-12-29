"""
Decoder-Only Transformer 完整实现
整合组件：
1. RoPE 位置编码 (rope_1227.py)
2. RMSNorm 归一化 (RMSNorm_251218.py)
3. Multi-Head Attention (mha_1227.py)
4. SwiGLU FFN (Qwen3 架构)
5. Top-P 解码
6. Decoder-Only 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 1. RoPE 位置编码 ====================
def compute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    计算旋转位置编码的频率
    Args:
        dim: 注意力头的维度
        seq_len: 序列长度
        theta: 基础频率参数
    Returns:
        freqs_cis: 复数形式的频率张量 [seq_len, dim//2]
    """
    # 计算频率基础值
    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # 位置索引
    position = torch.arange(seq_len, dtype=torch.float)
    # 外积得到角度
    angles = torch.outer(position, freqs_base)
    # 转换为复数形式（极坐标）
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    应用 RoPE 位置编码
    Args:
        xq: Query张量 [bsz, seq, num_heads, head_dim]
        xk: Key张量 [bsz, seq, num_heads, head_dim]
        freqs_cis: 频率张量 [seq_len, head_dim//2]
    Returns:
        xq_out, xk_out: 应用RoPE后的Q和K张量
    """
    # 转换为复数形式 [bsz, seq, num_heads, head_dim//2, 2]
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转换为复数张量 [bsz, seq, num_heads, head_dim//2]
    xq_complex = torch.view_as_complex(xq_reshaped)
    xk_complex = torch.view_as_complex(xk_reshaped)
    
    # 调整频率张量形状以匹配 [1, seq, 1, head_dim//2]
    seq = xq_complex.shape[1]
    freqs_cis = freqs_cis[:seq].reshape(1, seq, 1, -1)
    
    # 旋转操作（复数乘法）
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis
    
    # 转回实数并展平 [bsz, seq, num_heads, head_dim]
    xq_out = torch.view_as_real(xq_rotated).flatten(3)
    xk_out = torch.view_as_real(xk_rotated).flatten(3)
    
    # 恢复原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==================== 2. RMSNorm 归一化 ====================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    相比LayerNorm更简单高效，不需要计算均值和方差
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [bsz, seq_len, d_model]
        Returns:
            normalized: [bsz, seq_len, d_model]
        """
        # 计算RMS并归一化
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        # 应用可学习的缩放参数
        return x_norm * self.weight


# ==================== 3. Multi-Head Attention ====================
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # Q, K, V 投影矩阵
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        # 输出投影矩阵
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [bsz, seq, d_model]
            freqs_cis: RoPE频率张量 [seq, head_dim//2]
            mask: 注意力掩码 [bsz, 1, seq, seq]
        Returns:
            output: [bsz, seq, d_model]
        """
        bsz, seq, _ = x.shape
        
        # 线性投影并重塑为多头形式 [bsz, seq, num_heads, head_dim]
        q = self.W_Q(x).reshape(bsz, seq, self.num_heads, self.head_dim)
        k = self.W_K(x).reshape(bsz, seq, self.num_heads, self.head_dim)
        v = self.W_V(x).reshape(bsz, seq, self.num_heads, self.head_dim)
        
        # 应用 RoPE 位置编码
        if freqs_cis is not None:
            q, k = apply_rope(q, k, freqs_cis)
        
        # 转置为 [bsz, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数 [bsz, num_heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（因果掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和 [bsz, num_heads, seq, head_dim]
        output = torch.matmul(attn_weights, v)
        
        # 重塑回 [bsz, seq, d_model]
        output = output.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        
        # 输出投影
        output = self.W_O(output)
        
        return output


# ==================== 4. SwiGLU FFN (Qwen3 架构) ====================
class SwiGLU_FFN(nn.Module):
    """
    SwiGLU 前馈网络
    使用 Swish 激活函数和门控线性单元（GLU）
    架构：Gate_proj, Up_proj, Down_proj
    """
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 默认扩展4倍
        
        # 三个线性投影层
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [bsz, seq, d_model]
        Returns:
            output: [bsz, seq, d_model]
        """
        # SwiGLU: (Swish(gate) * up) @ down
        gate = F.silu(self.gate_proj(x))  # Swish(x) = x * sigmoid(x) = SiLU(x)
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        return output


# ==================== 5. Decoder Block ====================
class DecoderBlock(nn.Module):
    """
    单个 Decoder 层
    架构：RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    """
    def __init__(self, num_heads: int, d_model: int, d_ff: int = None, eps: float = 1e-5):
        super().__init__()
        # 注意力层前的归一化
        self.attn_norm = RMSNorm(d_model, eps)
        self.attention = MultiHeadAttention(num_heads, d_model)
        
        # FFN层前的归一化
        self.ffn_norm = RMSNorm(d_model, eps)
        self.ffn = SwiGLU_FFN(d_model, d_ff)
    
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [bsz, seq, d_model]
            freqs_cis: RoPE频率张量
            mask: 因果掩码
        Returns:
            output: [bsz, seq, d_model]
        """
        # 注意力子层（Pre-Norm）
        h = x + self.attention(self.attn_norm(x), freqs_cis, mask)
        
        # FFN子层（Pre-Norm）
        out = h + self.ffn(self.ffn_norm(h))
        
        return out


# ==================== 6. Decoder-Only Transformer ====================
class DecoderOnlyTransformer(nn.Module):
    """
    完整的 Decoder-Only Transformer 模型
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int = None,
        max_seq_len: int = 2048,
        eps: float = 1e-5,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 预计算RoPE频率
        head_dim = d_model // num_heads
        self.register_buffer(
            'freqs_cis',
            compute_freqs_cis(head_dim, max_seq_len, rope_theta)
        )
        
        # Decoder层堆叠
        self.layers = nn.ModuleList([
            DecoderBlock(num_heads, d_model, d_ff, eps)
            for _ in range(num_layers)
        ])
        
        # 最后的归一化层
        self.norm = RMSNorm(d_model, eps)
        
        # 输出层（language model head）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定（可选）
        # self.lm_head.weight = self.token_embedding.weight
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [bsz, seq]
            targets: [bsz, seq] 用于计算loss
        Returns:
            logits: [bsz, seq, vocab_size]
            loss: 如果提供targets则返回loss
        """
        bsz, seq = input_ids.shape
        
        # Token嵌入
        x = self.token_embedding(input_ids)  # [bsz, seq, d_model]
        
        # 生成因果掩码（下三角矩阵）
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).unsqueeze(0).unsqueeze(0)
        # [1, 1, seq, seq]
        
        # 获取对应序列长度的RoPE频率
        freqs_cis = self.freqs_cis[:seq]
        
        # 通过所有Decoder层
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        
        # 最终归一化
        x = self.norm(x)
        
        # 投影到词表空间
        logits = self.lm_head(x)  # [bsz, seq, vocab_size]
        
        # 计算loss（如果提供targets）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        使用 Top-P 采样生成文本
        Args:
            input_ids: [bsz, seq] 输入token序列
            max_new_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            top_p: nucleus sampling参数
            eos_token_id: 结束token ID
        Returns:
            generated: [bsz, seq + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # 如果序列超过最大长度，截断
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            # 前向传播
            logits, _ = self.forward(input_ids)
            
            # 取最后一个位置的logits
            logits = logits[:, -1, :] / temperature  # [bsz, vocab_size]
            
            # Top-P 采样
            next_token = top_p_sampling(logits, top_p)
            
            # 拼接到序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            
            # 检查是否生成结束token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids


# ==================== 7. Top-P 采样解码 ====================
def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """
    Top-P (Nucleus) 采样
    选择累积概率超过top_p的最小token集合进行采样
    
    Args:
        logits: [bsz, vocab_size] 未归一化的分数
        top_p: 累积概率阈值 (0, 1]
    Returns:
        next_token: [bsz] 采样得到的token
    """
    # 转换为概率分布
    probs = F.softmax(logits, dim=-1)  # [bsz, vocab_size]
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 计算累积概率
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过top_p的位置
    # 保留累积概率刚好超过top_p的token
    sorted_indices_to_remove = cumsum_probs > top_p
    
    # 保留第一个超过阈值的token（确保至少有一个token）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 创建掩码并过滤概率
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    probs = probs.masked_fill(indices_to_remove, 0.0)
    
    # 重新归一化
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # 从过滤后的分布中采样
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    
    return next_token


# ==================== 8. 示例使用 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模型配置
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 128
    
    # 创建模型
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    print("=" * 60)
    print("Decoder-Only Transformer 模型架构")
    print("=" * 60)
    print(f"词表大小: {vocab_size}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {num_heads}")
    print(f"每个头维度: {d_model // num_heads}")
    print(f"层数: {num_layers}")
    print(f"FFN维度: {d_ff}")
    print(f"最大序列长度: {max_seq_len}")
    print("=" * 60)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("=" * 60)
    
    # 1. 训练示例
    print("\n1. 训练模式测试")
    bsz = 4
    seq = 32
    
    # 随机生成输入和目标
    input_ids = torch.randint(0, vocab_size, (bsz, seq))
    targets = torch.randint(0, vocab_size, (bsz, seq))
    
    print(f"输入形状: {input_ids.shape}")
    
    # 前向传播
    logits, loss = model(input_ids, targets)
    
    print(f"输出logits形状: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 2. 生成示例
    print("\n2. 生成模式测试 (Top-P 采样)")
    model.eval()
    
    # 输入提示
    prompt_ids = torch.randint(0, vocab_size, (1, 10))
    print(f"提示序列长度: {prompt_ids.shape[1]}")
    
    # 生成文本
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_p=0.9
    )
    
    print(f"生成序列长度: {generated_ids.shape[1]}")
    print(f"生成的token数: {generated_ids.shape[1] - prompt_ids.shape[1]}")
    
    # 3. 组件测试
    print("\n3. 各组件独立测试")
    
    # RoPE测试
    print("\n(1) RoPE 位置编码测试")
    head_dim = 64
    seq_len = 32
    freqs = compute_freqs_cis(head_dim, seq_len)
    print(f"  频率张量形状: {freqs.shape}")
    
    xq = torch.randn(2, seq_len, 8, head_dim)
    xk = torch.randn(2, seq_len, 8, head_dim)
    xq_rot, xk_rot = apply_rope(xq, xk, freqs)
    print(f"  旋转前Q形状: {xq.shape}")
    print(f"  旋转后Q形状: {xq_rot.shape}")
    
    # RMSNorm测试
    print("\n(2) RMSNorm 归一化测试")
    x = torch.randn(2, 32, d_model)
    norm = RMSNorm(d_model)
    x_norm = norm(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {x_norm.shape}")
    print(f"  输入均值: {x.mean():.4f}, 方差: {x.var():.4f}")
    print(f"  输出均值: {x_norm.mean():.4f}, 方差: {x_norm.var():.4f}")
    
    # MHA测试
    print("\n(3) Multi-Head Attention 测试")
    x = torch.randn(2, 32, d_model)
    mha = MultiHeadAttention(num_heads, d_model)
    out = mha(x, freqs)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    
    # SwiGLU FFN测试
    print("\n(4) SwiGLU FFN 测试")
    x = torch.randn(2, 32, d_model)
    ffn = SwiGLU_FFN(d_model, d_ff)
    out = ffn(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    
    # Top-P采样测试
    print("\n(5) Top-P 采样测试")
    logits = torch.randn(2, vocab_size)
    sampled = top_p_sampling(logits, top_p=0.9)
    print(f"  Logits形状: {logits.shape}")
    print(f"  采样结果: {sampled}")
    print(f"  采样结果形状: {sampled.shape}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
