import torch
import torch.nn as nn

# RoPE旋转位置编码
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 核心：计算频率theta，注册为buffer（不参与训练）
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor):
        # 获取输入维度: [batch, seq_len, dim]
        bsz, seq, dim = x.shape
        # 生成位置序列 [0,1,...,seq-1]，自动对齐设备
        pos = torch.arange(seq, device=x.device)# [seq_len]
        
        # 计算正弦余弦位置因子
        # pos.unsqueeze(-1) -> [seq,1]; theta -> [dim/2]
        # 广播机制：① 从后向前补充维度：theta 向前补充一维：[1,dim/2];② 维度为 1 的自动扩展对齐 
        angles = pos.unsqueeze(-1) * self.theta  # [seq, dim//2]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # 拆分偶数/奇数维度，各自 [bsz, seq, dim//2]
        x1, x2 = x[..., ::2], x[..., 1::2]
        # 旋转公式，cos/sin 自动广播到 batch 维度
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # 拼接回原维度：x1_rot：[bsz, seq, dim//2]->[bsz, seq, dim//2,1]->[bsz, seq, dim//2,2]->[bsz, seq, dim]
        return torch.cat([x1_rot.unsqueeze(-1), x2_rot.unsqueeze(-1)], dim=-1).flatten(-2)

if __name__ == "__main__":
    batch, seq, dim = 32, 50, 64
    rope = RotaryPositionEmbedding(dim)
    x = torch.randn(batch, seq, dim)
    out = rope(x)
    
    print("输入shape:", x.shape)
    print("输出shape:", out.shape)