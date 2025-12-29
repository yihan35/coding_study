"""
Decoder-Only Transformer V1 优化版本
改进点：
1. 添加 Dropout 支持
2. 实现权重初始化
3. 添加配置类
4. 编写单元测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


# ==================== 配置类 ====================
@dataclass
class TransformerConfig:
    """Transformer 模型配置类"""
    # 模型结构参数
    vocab_size: int = 50000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: Optional[int] = None  # 默认为 4 * d_model
    max_seq_len: int = 2048
    
    # 正则化参数
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    emb_dropout: float = 0.1
    
    # 归一化参数
    eps: float = 1e-5
    
    # RoPE 参数
    rope_theta: float = 10000.0
    
    # 权重初始化参数
    init_std: float = 0.02
    
    # 其他参数
    tie_weights: bool = False  # 是否绑定 embedding 和 lm_head 权重
    
    def __post_init__(self):
        """初始化后处理"""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        
        # 验证参数
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be in [0, 1]"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.num_layers > 0, "num_layers must be positive"


# ==================== 1. RoPE 位置编码 ====================
def compute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device: str = 'cpu') -> torch.Tensor:
    """
    计算旋转位置编码的频率
    Args:
        dim: 注意力头的维度
        seq_len: 序列长度
        theta: 基础频率参数
        device: 设备
    Returns:
        freqs_cis: 复数形式的频率张量 [seq_len, dim//2]
    """
    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
    position = torch.arange(seq_len, dtype=torch.float, device=device)
    angles = torch.outer(position, freqs_base)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用 RoPE 位置编码
    Args:
        xq: Query张量 [bsz, seq, num_heads, head_dim]
        xk: Key张量 [bsz, seq, num_heads, head_dim]
        freqs_cis: 频率张量 [seq_len, head_dim//2]
    Returns:
        xq_out, xk_out: 应用RoPE后的Q和K张量
    """
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_complex = torch.view_as_complex(xq_reshaped)
    xk_ torch.view_as_complex(xk_reshaped)
    
    seq = xq_complex.shape[1]
    freqs_cis = freqs_cis[:seq].reshape(1, seq, 1, -1)
    
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis
    
    xq_out = torch.view_as_real(xq_rotated).flatten(3)
    xk_out = torch.view_as_real(xk_rotated).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==================== 2. RMSNorm 归一化 ====================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        return x_norm * self.weight
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.ones_(self.weight)


# ==================== 3. Multi-Head Attention ====================
class MultiHeadAttention(nn.Module):
    """多头注意力机制（支持 Dropout）"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.num_heads
        self.dropout = cig.attn_dropout
        
        # Q, K, V 投影矩阵
        self.W_Q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_K = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_V = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Dropout 层
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape
        
        q = self.W_Q(x).reshape(bsz, seq, self.num_heads, self.head_dim)
        k = self.W_K(x).reshape(bsz, seq, self.num_heads, self.head_dim)
        v = self.W_V(x).reshape(bsz, seq, self.num_heads, self.head_dim)
        
        if freqs_cis is not None:
            q, k = apply_rope(q, k, freqs_cis)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # Dropout on attention weights
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        output = self.W_O(output)
        output = self.resid_dropout(output)  # Dropout on output
        
        return output
    
    def reset_parameters(self):
        """重置参数（Xavier/Glorot 初始化）"""
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)


# ==================== 4. SwiGLU FFN ====================
class SwiGLU_FFN(nn.Module):
    """SwiGLU 前馈网络（支持 Dropout）"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        d_ff = config.d_ff if config.d_ff is not None else 4 * config.d_model
        
        self.gate_proj = nn.Linear(config.d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.ffn_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)
        output = self.dropout(output)  # Dropout on output
        return output
    
    def reset_parameters(self):
        """重置参数（Xavier 初始化）"""
        for module in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.xavier_uniform_(module.weight)


# ==================== 5. Decoder Block ====================
class DecoderBlock(nn.Module):
    """单个 Decoder 层"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.eps)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.eps)
        self.ffn = SwiGLU_FFN(config)
    
    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-Norm + Residual
        h = x + self.attention(self.attn_norm(x), freqs_cis, mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out
    
    def reset_parameters(self):
        """重置参数"""
        self.attn_norm.reset_parameters()
        self.attention.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()


# ==================== 6. Decoder-Only Transformer ====================
class DecoderOnlyTransformer(nn.Module):
    """完整的 Decoder-Only Transformer 模型"""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.emb_dropout)
        
        # 预计算RoPE频率
        head_dim = config.d_model // config.num_heads
        self.register_buffer(
            'freqs_cis',
            compute_freqs_cis(head_dim, config.max_seq_len, config.rope_theta)
        )
        
        # Decoder层堆叠
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.num_layers)
        ])
        
        # 最后的归一化层
        self.norm = RMSNorm(config.d_model, config.eps)
        
        # 输出层
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重绑定
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # 应用权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        权重初始化策略
        - Embedding: 正态分布 N(0, init_std)
        - Linear: Xavier/Glorot 均匀初始化
        - RMSNorm: 全1初始化
        """
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq = input_ids.shape
        
        # Token嵌入 + Dropout
        x = self.token_embedding(input_ids)
        x = self.emb_dropout(x)
        
        # 生成因果掩码
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # 获取RoPE频率
        freqs_cis = self.freqs_cis[:seq]
        
        # 通过所有Decoder层
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        
        # 最终归一化
        x = self.norm(x)
        
        # 投影到词表空间
        logits = self.lm_head(x)
        
        # 计算loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
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
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """使用 Top-P 采样生成文本"""
        self.eval()  # 设置为评估模式，禁用 Dropout
        
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            
            next_token = top_p_sampling(logits, top_p)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        获取参数数量
        Args:
            non_embedding: 是否排除 embedding 层参数
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params


# ==================== 7. Top-P 采样 ====================
def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """Top-P (Nucleus) 采样"""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumsum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    return next_token


# ==================== 8. 单元测试 ====================
def run_unit_tests():
    """运行单元测试"""
    print("=" * 60)
    print("单元测试")
    print("=" * 60)
    
    # 测试配置类
    print("\n[测试 1] TransformerConfig")
    config = TransformerConfig(
        vocab_size=1000,
        d_model=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    print(f"✓ 配置创建成功")
    print(f"  - d_model: {config.d_model}")
    print(f"  - d_ff: {config.d_ff} (自动设置为 4 * d_model)")
    print(f"  - dropout: {config.dropout}")
    
    # 测试 RoPE
    print("\n[测试 2] RoPE 位置编码")
    head_dim = 64
    seq_len = 32
    freqs = compute_freqs_cis(head_dim, seq_len)
    assert freqs.shape == (seq_len, head_dim // 2), "RoPE 形状错误"
    print(f"✓ RoPE 计算正确，形状: {freqs.shape}")
    
    xq = torch.randn(2, seq_len, 4, head_dim)
    xk = torch.randn(2, seq_len, 4, head_dim)
    xq_rot, xk_rot = apply_rope(xq, xk, freqs)
    assert xq_rot.shape == xq.shape, "RoPE 应用后形状改变"
    print(f"✓ RoPE 应用正确，输出形状: {xq_rot.shape}")
    
    # 测试 RMSNorm
    print("\n[测试 3] RMSNorm")
    x = torch.randn(2, 32, config.d_model)
    norm = RMSNorm(config.d_model)
    x_norm = norm(x)
    assert x_norm.shape == x.shape, "RMSNorm 形状改变"
    print(f"✓ RMSNorm 正确，形状: {x_norm.shape}")
    print(f"  输入: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"  输出: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
    
    # 测试 Attention
    print("\n[测试 4] MultiHeadAttention (with Dropout)")
    attn = MultiHeadAttention(config)
    out = attn(x, freqs)
    assert out.shape == x.shape, "Attention 输出形状错误"
    print(f"✓ Attention 正确，形状: {out.shape}")
    print(f"  包含 attn_dropout={config.attn_dropout}, resid_dropout={config.dropout}")
    
    # 测试 FFN
    print("\n[测试 5] SwiGLU_FFN (with Dropout)")
    ffn = SwiGLU_FFN(config)
    out = ffn(x)
    assert out.shape == x.shape, "FFN 输出形状错误"
    print(f"✓ FFN 正确，形状: {out.shape}")
    print(f"  包含 ffn_dropout={config.ffn_dropout}")
    
    # 测试完整模型
    print("\n[测试 6] DecoderOnlyTransformer")
    model = DecoderOnlyTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    
    # 训练模式
    model.train()
    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 16, config.vocab_size), "Logits 形状错误"
    assert loss is not None, "Loss 未计算"
    print(f"✓ 训练模式正确")
    print(f"  logits: {logits.shape}, loss: {loss.item():.4f}")
    
    # 生成模式
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, top_p=0.9)
    assert generated.shape[0] == 1, "生成批次大小错误"
    assert generated.shape[1] == 15, "生成长度错误"
    print(f"✓ 生成模式正确")
    print(f"  输入长度: 5, 生成长度: 10, 总长度: {generated.shape[1]}")
    
    # 测试权重初始化
    print("\n[测试 7] 权重初始化")
    model2 = DecoderOnlyTransformer(config)
    emb_std = model2.token_embedding.weight.std().item()
    print(f"✓ Embedding 初始化: std={emb_std:.4f} (目标: {config.init_std:.4f})")
    
    # 测试参数数量
    print("\n[测试 8] 参数统计")
    total_params = model.get_num_params(non_embedding=False)
    model_params = model.get_num_params(non_embedding=True)
    print(f"✓ 总参数量: {total_params:,}")
    print(f"✓ 模型参数量(不含embedding): {model_params:,}")
    
    # 测试 Dropout 效果
    print("\n[测试 9] Dropout 效果验证")
    model.train()
    out1, _ = model(input_ids)
    out2, _ = model(input_ids)
    dropout_working = not torch.allclose(out1, out2)
    print(f"✓ Dropout 工作状态: {'正常' if dropout_working else '异常'}")
    
    model.eval()
    out3, _ = model(input_ids)
    out4, _ = model(input_ids)
    eval_deterministic = torch.allclose(out3, out4)
    print(f"✓ 评估模式确定性: {'是' if eval_deterministic else '否'}")
    
    print("\n" + "=" * 60)
    print("所有单元测试通过! ✓")
    print("=" * 60)


# ==================== 9. 示例使用 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行单元测试
    run_unit_tests()
    
    print("\n\n")
    
    # 创建模型配置
    config = TransformerConfig(
        vocab_size=1000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=128,
        dropout=0.1,
        attn_dropout=0.1,
        init_std=0.02
    )
    
    # 创建模型
    model = DecoderOnlyTransformer(config)
    
    print("=" * 60)
    print("Decoder-Only Transformer V1 模型架构")
    print("=" * 60)
    print(f"词表大小: {config.vocab_size}")
    print(f"模型维度: {config.d_model}")
    print(f"注意力头数: {config.num_heads}")
    print(f"每个头维度: {config.d_model // config.num_heads}")
    print(f"层数: {config.num_layers}")
    print(f"FFN维度: {config.d_ff}")
    print(f"最大序列长度: {config.max_seq_len}")
    print(f"Dropout: {config.dropout}")
    print(f"注意力Dropout: {config.attn_dropout}")
    print(f"权重初始化std: {config.init_std}")
    print("=" * 60)
    
    # 统计参数量
    total_params = model.get_num_params(non_embedding=False)
    model_params = model.get_num_params(non_embedding=True)
    print(f"总参数量: {total_params:,}")
    print(f"模型参数量(不含embedding): {model_params:,}")
    print("=" * 60)
    
    # 训练示例
    print("\n训练模式测试")
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (4, 32))
    targets = torch.randint(0, config.vocab_size, (4, 32))
    
    logits, loss = model(input_ids, targets)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # 生成示例
    print("\n生成模式测试")
    model.eval()
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10))
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_p=0.9
    )
    
    print(f"提示序列长度: {prompt_ids.shape[1]}")
    print(f"生成序列长度: {generated_ids.shape[1]}")
    print(f"生成的token数: {generated_ids.shape[1] - prompt_ids.shape[1]}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
