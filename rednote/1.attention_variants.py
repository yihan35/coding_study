import torch
import math
import torch.nn as nn


# ─────────────────────────────────────────────
# 1. Scaled Dot-Product Attention（单头）
# ─────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x):
        bsz, seq, d_model = x.shape
        xq = self.Q(x)  # (bsz, seq, d_model)
        xk = self.K(x)
        xv = self.V(x)
        att = torch.matmul(xq, xk.transpose(-1, -2))           # (bsz, seq, seq)
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att_norm = torch.softmax(att / math.sqrt(d_model), dim=-1)
        out = torch.matmul(att_norm, xv)                        # (bsz, seq, d_model)
        return self.O(out)


# ─────────────────────────────────────────────
# 2. Multi-Head Attention（MHA）
# ─────────────────────────────────────────────
class MHA(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x):
        bsz, seq, d_model = x.shape
        head = self.head
        d = d_model // head
        xq = self.Q(x).reshape(bsz, seq, head, d).transpose(1, 2)  # (bsz, head, seq, d)
        xk = self.K(x).reshape(bsz, seq, head, d).transpose(1, 2)
        xv = self.V(x).reshape(bsz, seq, head, d).transpose(1, 2)
        att = torch.matmul(xq, xk.transpose(-1, -2))                # (bsz, head, seq, seq)
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att_norm = torch.softmax(att / math.sqrt(d), dim=-1)
        out = torch.matmul(att_norm, xv).transpose(1, 2).contiguous().view(bsz, seq, d_model)
        return self.O(out)


# ─────────────────────────────────────────────
# 3. MHA + KV Cache（用于自回归推理）
# ─────────────────────────────────────────────
class MHA_KVCache(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        bsz, seq, d_model = x.shape
        head = self.head
        d = d_model // head
        xq = self.Q(x).reshape(bsz, seq, head, d).transpose(1, 2)
        xk = self.K(x).reshape(bsz, seq, head, d).transpose(1, 2)
        xv = self.V(x).reshape(bsz, seq, head, d).transpose(1, 2)
        if past_kv is not None:
            past_k, past_v = past_kv
            xk = torch.cat([past_k, xk], dim=2)   # 历史 K 拼接新 K
            xv = torch.cat([past_v, xv], dim=2)
        present_kv = (xk, xv)
        total_seq = xk.shape[2]
        att = torch.matmul(xq, xk.transpose(-1, -2))            # (bsz, head, seq, total_seq)
        mask = torch.tril(torch.ones(total_seq, total_seq, device=x.device)).bool()
        mask = mask[-seq:, :]                                    # 只取当前 seq 对应的行
        att = att.masked_fill(~mask, float('-inf'))
        att_norm = torch.softmax(att / math.sqrt(d), dim=-1)
        out = torch.matmul(att_norm, xv).transpose(1, 2).contiguous().view(bsz, seq, d_model)
        return self.O(out), present_kv


# ─────────────────────────────────────────────
# 4. Multi-Query Attention（MQA）
#    Q 有 h 个头，K/V 只有 1 个头，所有 Q 共享
# ─────────────────────────────────────────────
class MQA(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        d = d_model // head
        self.Q = nn.Linear(d_model, d_model)    # 输出: h 个头
        self.K = nn.Linear(d_model, d)          # 输出: 1 个头
        self.V = nn.Linear(d_model, d)          # 输出: 1 个头
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x):
        bsz, seq, d_model = x.shape
        head = self.head
        d = d_model // head
        xq = self.Q(x).reshape(bsz, seq, head, d).transpose(1, 2)  # (bsz, head, seq, d)
        xk = self.K(x).reshape(bsz, seq, 1, d).transpose(1, 2)     # (bsz, 1, seq, d)
        xv = self.V(x).reshape(bsz, seq, 1, d).transpose(1, 2)     # (bsz, 1, seq, d)
        # 广播: xk/xv 的 head 维度为 1，自动广播到 h 个 Q 头
        att = torch.matmul(xq, xk.transpose(-1, -2))                # (bsz, head, seq, seq)
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att_norm = torch.softmax(att / math.sqrt(d), dim=-1)
        out = torch.matmul(att_norm, xv).transpose(1, 2).contiguous().view(bsz, seq, d_model)
        return self.O(out)


# ─────────────────────────────────────────────
# 5. Grouped-Query Attention（GQA）
#    Q 有 h 个头，K/V 有 g 个头，每组 h//g 个 Q 共享一对 K/V
#    g=1 退化为 MQA，g=h 退化为 MHA
# ─────────────────────────────────────────────
class GQA(nn.Module):
    def __init__(self, d_model, head, kv_head):
        super().__init__()
        assert head % kv_head == 0, "head 必须能被 kv_head 整除"
        self.d_model = d_model
        self.head = head
        self.kv_head = kv_head
        self.group = head // kv_head   # 每组的 Q 头数
        d = d_model // head
        self.Q = nn.Linear(d_model, d_model)            # h 个 Q 头
        self.K = nn.Linear(d_model, d * kv_head)        # g 个 K 头
        self.V = nn.Linear(d_model, d * kv_head)        # g 个 V 头
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x):
        bsz, seq, d_model = x.shape
        head = self.head
        kv_head = self.kv_head
        group = self.group
        d = d_model // head
        xq = self.Q(x).reshape(bsz, seq, head, d).transpose(1, 2)      # (bsz, h, seq, d)
        xk = self.K(x).reshape(bsz, seq, kv_head, d).transpose(1, 2)   # (bsz, kv_h, seq, d)
        xv = self.V(x).reshape(bsz, seq, kv_head, d).transpose(1, 2)   # (bsz, kv_h, seq, d)
        # 将 K/V 扩展到 h 个头：每个 KV 头重复 group 次
        xk = xk.repeat_interleave(group, dim=1)                         # (bsz, h, seq, d)
        xv = xv.repeat_interleave(group, dim=1)
        att = torch.matmul(xq, xk.transpose(-1, -2))                    # (bsz, h, seq, seq)
        mask = torch.tril(torch.ones(seq, seq, device=x.device)).bool()
        att = att.masked_fill(~mask, float('-inf'))
        att_norm = torch.softmax(att / math.sqrt(d), dim=-1)
        out = torch.matmul(att_norm, xv).transpose(1, 2).contiguous().view(bsz, seq, d_model)
        return self.O(out)


# ─────────────────────────────────────────────
# 6. Multi-Head Latent Attention（MLA）
#    DeepSeek-V2 提出，将 K/V 压缩到低维潜在向量后缓存
#    推理时只需缓存潜在向量 c，显存大幅减少
# ─────────────────────────────────────────────
class MLA(nn.Module):
    def __init__(self, d_model, head, d_c):
        super().__init__()
        # d_c: KV 压缩维度，远小于 d_model（论文中约为 d_model 的 1/8 ~ 1/4）
        self.d_model = d_model
        self.head = head
        self.d_c = d_c
        d = d_model // head
        self.Q = nn.Linear(d_model, d_model)
        self.kv_down = nn.Linear(d_model, d_c)             # 压缩：输入 → 潜在向量 c
        self.k_up   = nn.Linear(d_c, d_model)              # 还原：c → K
        self.v_up   = nn.Linear(d_c, d_model)              # 还原：c → V
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x, past_c=None):
        bsz, seq, d_model = x.shape
        head = self.head
        d = d_model // head
        xq = self.Q(x).reshape(bsz, seq, head, d).transpose(1, 2)  # (bsz, head, seq, d)
        # 压缩 K/V 为低维潜在向量（这是 KV Cache 实际缓存的内容）
        c = self.kv_down(x)                                          # (bsz, seq, d_c)
        if past_c is not None:
            c = torch.cat([past_c, c], dim=1)                        # 拼接历史潜在向量
        present_c = c
        total_seq = c.shape[1]
        # 推理时从潜在向量还原 K/V
        xk = self.k_up(c).reshape(bsz, total_seq, head, d).transpose(1, 2)
        xv = self.v_up(c).reshape(bsz, total_seq, head, d).transpose(1, 2)
        att = torch.matmul(xq, xk.transpose(-1, -2))                # (bsz, head, seq, total_seq)
        mask = torch.tril(torch.ones(total_seq, total_seq, device=x.device)).bool()
        mask = mask[-seq:, :]
        att = att.masked_fill(~mask, float('-inf'))
        att_norm = torch.softmax(att / math.sqrt(d), dim=-1)
        out = torch.matmul(att_norm, xv).transpose(1, 2).contiguous().view(bsz, seq, d_model)
        return self.O(out), present_c


# ─────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────
bsz, seq, d_model = 2, 4, 64
head    = 8
kv_head = 2           # GQA: 2 组，每组 4 个 Q 头共享一对 KV
d_c     = 16          # MLA: 压缩维度

x = torch.randn(bsz, seq, d_model)

print("1. Attention:      ", Attention(d_model)(x).shape)
print("2. MHA:            ", MHA(d_model, head)(x).shape)

mha_kv = MHA_KVCache(d_model, head)
out, past_kv = mha_kv(x)
print("3. MHA_KVCache:    ", out.shape)
out, past_kv = mha_kv(torch.randn(bsz, 1, d_model), past_kv)
print("   MHA_KVCache decode:", out.shape, " kv_seq:", past_kv[0].shape[2])

print("4. MQA:            ", MQA(d_model, head)(x).shape)
print("5. GQA:            ", GQA(d_model, head, kv_head)(x).shape)

mla = MLA(d_model, head, d_c)
out, past_c = mla(x)
print("6. MLA:            ", out.shape)
out, past_c = mla(torch.randn(bsz, 1, d_model), past_c)
print("   MLA decode:     ", out.shape, " c_seq:", past_c.shape[1])


import torch
import torch.nn as nn
class RMS(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self,x):
        x_new = torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)*x
        return x_new * self.weight
    



import torch
import torch.nn as nn
class LayNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True)
        x_2 = (x-mean)/torch.sqrt(var+self.eps)
        x_out = x_2 * self.alpha + self.beta
        return x_out
    



import math
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self,d_model,eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d_model))
    def forward(self,x):
        return x* torch.rsqrt(x.power(2).mean(dim=-1,keepdim = True)+self.eps) * self.w

class LayerNorm(nn.Module):
    def __init__(self,):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
