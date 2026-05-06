import torch
import math
import torch.nn as nn

class MHAWithKVCache(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.dim = dim
        self.head = head
        self.d = dim // head  # 每个注意力头的维度
        # 确保维度能被头数整除（防御性检查）
        assert dim % head == 0, "dim must be divisible by head"
        
        # Q/K/V/O 线性投影层（与基础MHA一致）
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.O = nn.Linear(dim, dim)
    
    def forward(self, x, past_kv=None):
        """
        前向传播（支持KV Cache的增量解码）
        Args:
            x: 输入张量，shape [bsz, seq_len, dim]
               - 首次计算（无cache）：seq_len可为任意长度
               - 增量解码（有cache）：seq_len=1（每次仅输入1个token）
            past_kv: 缓存的历史K/V，tuple(past_k, past_v)
                     - past_k: [bsz, head, past_seq_len, d]
                     - past_v: [bsz, head, past_seq_len, d]
                     - None表示无缓存
        
        Returns:
            output: 注意力输出，shape [bsz, seq_len, dim]
            present_kv: 更新后的缓存，tuple(new_k, new_v)
                        - new_k: [bsz, head, total_seq_len, d]
                        - new_v: [bsz, head, total_seq_len, d]
        """
        bsz, seq_len, dim = x.shape
        
        # 1. 计算Q/K/V并调整维度 [bsz, seq_len, dim] → [bsz, head, seq_len, d]
        q = self.Q(x).reshape(bsz, seq_len, self.head, self.d).transpose(1, 2)
        k = self.K(x).reshape(bsz, seq_len, self.head, self.d).transpose(1, 2)
        v = self.V(x).reshape(bsz, seq_len, self.head, self.d).transpose(1, 2)
        
        # 2. 处理KV Cache：拼接历史K/V和当前K/V
        if past_kv is not None:
            past_k, past_v = past_kv
            # 在seq_len维度（dim=2）拼接，总长度=历史长度+当前长度
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # 保存更新后的缓存（供下一次增量解码使用）
        present_kv = (k, v)
        
        # 3. 计算注意力权重（缩放点积注意力）
        attn_scores = torch.matmul(q, k.transpose(-1, -2))  # [bsz, head, seq_len, total_seq_len]
        attn_scores = attn_scores / math.sqrt(self.d)  # 缩放防止梯度消失
        
        # 4. Softmax归一化注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 5. 加权求和V，调整维度并通过输出层
        attn_output = torch.matmul(attn_weights, v)  # [bsz, head, seq_len, d]
        # 维度还原：[bsz, head, seq_len, d] → [bsz, seq_len, dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        output = self.O(attn_output)
        
        return output, present_kv

# 测试代码
if __name__ == "__main__":
    # 超参数（与你的示例保持一致）
    bsz, seq_len, dim = 2, 4, 10
    head = 2
    
    # 随机输入张量
    x = torch.randn(bsz, seq_len, dim)
    print(f"初始输入形状: {x.shape}")  # [2, 4, 10]
    
    # 初始化带KV Cache的MHA
    mha = MHAWithKVCache(dim, head)
    
    # 测试1：首次前向传播（无缓存）
    output_first, past_kv = mha(x)
    print(f"首次输出形状: {output_first.shape}")  # [2, 4, 10]
    print(f"缓存K形状: {past_kv[0].shape}")        # [2, 2, 4, 5] (bsz, head, seq_len, d=5)
    print(f"缓存V形状: {past_kv[1].shape}")        # [2, 2, 4, 5]
    
    # 测试2：增量解码（有缓存，输入单个token）
    x_next = torch.randn(bsz, 1, dim)  # 新输入：每个样本1个token
    print(f"\n增量输入形状: {x_next.shape}")       # [2, 1, 10]
    output_next, past_kv_new = mha(x_next, past_kv=past_kv)
    print(f"增量输出形状: {output_next.shape}")    # [2, 1, 10]
    print(f"更新后缓存K形状: {past_kv_new[0].shape}")  # [2, 2, 5, 5] (4+1=5)
    print(f"更新后缓存V形状: {past_kv_new[1].shape}")  # [2, 2, 5, 5]
    
    # 验证：连续增量解码4次，缓存长度应等于初始seq_len
    past_kv_reset = None
    for i in range(4):
        x_step = torch.randn(bsz, 1, dim)
        _, past_kv_reset = mha(x_step, past_kv=past_kv_reset)
        print(f"\n第{i+1}步增量解码后，缓存K长度: {past_kv_reset[0].shape[2]}")
    assert past_kv_reset[0].shape[2] == 4, "缓存长度验证失败！"
    print("\n✅ 增量解码缓存逻辑验证通过")