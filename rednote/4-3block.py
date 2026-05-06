import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, dim]
        # 计算每个 token 向量的均方根，在最后一个维度上
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ============================================================
# RoPE 旋转位置编码
# ============================================================
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor):
        bsz, seq, dim = x.shape
        pos    = torch.arange(seq, device=x.device)
        angles = pos.unsqueeze(-1) * self.theta          # [seq, dim//2]
        cos    = torch.cos(angles)
        sin    = torch.sin(angles)
        x1, x2   = x[..., ::2], x[..., 1::2]
        x1_rot   = x1 * cos - x2 * sin
        x2_rot   = x1 * sin + x2 * cos
        return torch.cat([x1_rot.unsqueeze(-1), x2_rot.unsqueeze(-1)], dim=-1).flatten(-2)


# ============================================================
# GQA Attention
# ============================================================
class GQAttention(nn.Module):
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须是 num_kv_heads 的整数倍"
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.G            = num_q_heads // num_kv_heads   # 分组数
        self.head_dim     = d_model // num_q_heads

        # QKV 投影矩阵（可学习）
        self.Wq = nn.Linear(d_model, num_q_heads  * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # Q 和 K 各自施加 RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor):
        bsz, seq, d_model = x.shape

        # 1. 投影得到 Q K V
        Q = self.Wq(x)   # [bsz, seq, num_q_heads  * head_dim]
        K = self.Wk(x)   # [bsz, seq, num_kv_heads * head_dim]
        V = self.Wv(x)   # [bsz, seq, num_kv_heads * head_dim]

        # 2. 分头
        Q = Q.view(bsz, seq, self.num_q_heads,  self.head_dim)
        K = K.view(bsz, seq, self.num_kv_heads, self.head_dim)
        V = V.view(bsz, seq, self.num_kv_heads, self.head_dim)

        # 3. 对每个头单独施加 RoPE
        # rope.forward 接收 [bsz, seq, head_dim]，所以在头维度上遍历
        Q = torch.stack([self.rope(Q[:, :, h, :]) for h in range(self.num_q_heads)],  dim=2)
        K = torch.stack([self.rope(K[:, :, h, :]) for h in range(self.num_kv_heads)], dim=2)

        # 4. GQA：将 K V 在头维度上复制 G 次，对齐 Q 的头数
        # [bsz, seq, num_kv_heads, head_dim] -> [bsz, seq, num_q_heads, head_dim]
        K = K.repeat_interleave(self.G, dim=2)
        V = V.repeat_interleave(self.G, dim=2)

        # 5. 转置便于矩阵乘法：[bsz, num_q_heads, seq, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 6. Attention score + causal mask
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # [bsz, heads, seq, seq]
        # 因果掩码：下三角为 0，上三角填 -inf，防止看到未来 token
        mask   = torch.triu(torch.full((seq, seq), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask
        attn   = F.softmax(scores, dim=-1)

        # 7. 加权求和 + 拼接多头 + 输出投影
        out = torch.matmul(attn, V)                              # [bsz, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(bsz, seq, d_model)
        return self.Wo(out)


# ============================================================
# FFN（SwiGLU）
# 两路并行：Gate 路经 SiLU 激活，Up 路线性升维，逐元素相乘后降维
# ============================================================
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        # 门控路：升维后经 SiLU 激活
        self.Wgate = nn.Linear(d_model, d_ffn, bias=False)
        # Up 路：直接线性升维
        self.Wup   = nn.Linear(d_model, d_ffn, bias=False)
        # Down 路：降回 d_model
        self.Wdown = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [bsz, seq, d_model]
        gate = F.silu(self.Wgate(x))   # [bsz, seq, d_ffn]，门控值
        up   = self.Wup(x)             # [bsz, seq, d_ffn]，线性升维
        return self.Wdown(gate * up)   # 逐元素相乘后降维


# ============================================================
# MoE（Mixture of Experts）
# 多个 FFN 专家，Router 为每个 token 选 top-k 个专家加权融合
# ============================================================
class MoE(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, num_experts: int, top_k: int):
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k       = top_k

        # 每个专家是一个独立的 SwiGLU FFN
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, d_ffn) for _ in range(num_experts)
        ])
        # Router：线性层，为每个 token 输出各专家的路由得分
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [bsz, seq, d_model]
        bsz, seq, d_model = x.shape
        # 将 token 展平方便路由：[bsz*seq, d_model]
        x_flat = x.view(-1, d_model)

        # 1. Router 打分，选出 top-k 专家
        router_logits  = self.router(x_flat)                         # [N, num_experts]
        router_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=-1), self.top_k, dim=-1
        )                                                             # 各 [N, top_k]
        # 对选中的权重重新归一化，使其和为 1
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        # 2. 每个 token 加权融合 top-k 专家的输出
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            # 取第 k 个专家索引和对应权重
            idx     = expert_indices[:, k]        # [N]
            weights = router_weights[:, k]        # [N]
            # 按专家分组计算，避免对每个 token 单独调用
            for e in range(self.num_experts):
                mask = (idx == e)                 # 哪些 token 路由到专家 e
                if mask.any():
                    expert_out       = self.experts[e](x_flat[mask].unsqueeze(0))
                    output[mask]    += weights[mask].unsqueeze(-1) * expert_out.squeeze(0)

        return output.view(bsz, seq, d_model)


# ============================================================
# 完整 Transformer Block（Pre-Norm + 残差连接）
# 支持普通 FFN 或 MoE 两种模式
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:      int,
        num_q_heads:  int,
        num_kv_heads: int,
        d_ffn:        int,
        use_moe:      bool = False,
        num_experts:  int  = 8,
        top_k:        int  = 2,
    ):
        super().__init__()
        # Pre-Norm：在 Attention 和 FFN 之前各做一次 RMSNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.attn = GQAttention(d_model, num_q_heads, num_kv_heads)

        # FFN 模式：普通 SwiGLU 或 MoE
        if use_moe:
            self.ffn = MoE(d_model, d_ffn, num_experts, top_k)
        else:
            self.ffn = SwiGLUFFN(d_model, d_ffn)

    def forward(self, x: torch.Tensor):
        # Attention 子层：Pre-Norm + 残差
        x = x + self.attn(self.norm1(x))
        # FFN 子层：Pre-Norm + 残差
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 极简测试
# ============================================================
if __name__ == "__main__":
    bsz, seq, d_model = 2, 16, 128
    x = torch.randn(bsz, seq, d_model)

    print("=" * 40)
    print("普通 FFN Block")
    block_ffn = TransformerBlock(
        d_model=d_model, num_q_heads=8, num_kv_heads=2, d_ffn=256
    )
    out = block_ffn(x)
    print("输入 shape:", x.shape)
    print("输出 shape:", out.shape)

    print("=" * 40)
    print("MoE Block（8 专家，top-2）")
    block_moe = TransformerBlock(
        d_model=d_model, num_q_heads=8, num_kv_heads=2, d_ffn=256,
        use_moe=True, num_experts=8, top_k=2
    )
    out_moe = block_moe(x)
    print("输入 shape:", x.shape)
    print("输出 shape:", out_moe.shape)