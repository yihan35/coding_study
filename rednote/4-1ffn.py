import torch
import torch.nn as nn
import torch.nn.functional as F

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

