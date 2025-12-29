import torch
import torch.nn as nn

# 位置编码
def computer_freqs_cis(dim,seq_len,theta=10000.0):
    # [dim/2]
    freq_base = 1.0 / (theta ** (torch.range(0,dim,2)/dim))
    # [seq_len]
    position = torch.arange(0,seq_len,dtype=torch.float32)
    # [seq_len,dim/2]
    angles = torch.outer(position,freq_base)
    out = torch.polar(torch.ones_like(angles),angles)
    return out
def apply_rope(xq,xk,freqs_cis):
    # xq:[bsz,seq_len,num_head,head_dim]
    # freqs_cis:[seq_len,dim/2]
    bsz,seq_len,num_head,head_dim = xq.shape
    # freqs_cis:[1,seq_len,1,dim/2]
    freqs_cis = freqs_cis.reshape(1,seq_len,1,-1)
    # xq_1:[bsz,seq_len,num_head,head_dim/2,2]
    xq_1 = xq.float().reshape(*xq.shape[:-1],-1,2)
    xk_1 = xk.float().reshape(*xk.shape[:-1],-1,2)
    # xq_2:[bsz,seq_len,num_head,head_dim/2]
    xq_2 = torch.view_as_complex(xq_1)
    xk_2 = torch.view_as_complex(xk_1)
    # xq_rope:[bsz,seq_len,num_head,head_dim/2]
    xq_rope = xq_2 * freqs_cis
    xk_rope = xk_2 * freqs_cis
    # xq_3:[bsz,seq_len,num_head,head_dim/2,2]
    xq_3 = torch.view_as_real(xq_rope)
    xk_3 = torch.view_as_real(xk_rope)
    # xq_out:[bsz,seq_len,num_head,head_dim]
    xq_out = xq_3.flatten(3)
    xk_out = xk_3.flatten(3)
    return xq_out.type_as(xq),xk_out.type_as(xk)

# RMSNorm 
class RMSNorm(nn.Module):
    def __init__(self,d_model,eps = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self,x):
        # 为什么要keepdim，例如张量 x 形状为[2, 3]，计算方差后默认形状为[2]，必须要keepdim = True 保证形状为[2,1]
        # variance.shape [bsz,seq_len,1]
        variance = x.pow(2).mean(dim=-1,keepdim = True)
        rms  = torch.rsqrt(variance+self.eps)
        # x.shape [bsz,seq_len,d]
        return x * rms * self.weight
