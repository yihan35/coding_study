import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.d_k = d_model // num_head
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        参数:
            x: [bsz, seq, d_model]
            mask: [bsz, 1, 1, seq] 或 [bsz, 1, seq, seq]
        """
        bsz, seq, d = x.shape  # x: [bsz, seq, d_model]
        
        # Step 1: 线性投影 + 分成多头
        q = self.W_Q(x).reshape(bsz, seq, self.num_head, self.d_k)  # [bsz, seq, num_head, d_k]
        k = self.W_K(x).reshape(bsz, seq, self.num_head, self.d_k)  # [bsz, seq, num_head, d_k]
        v = self.W_V(x).reshape(bsz, seq, self.num_head, self.d_k)  # [bsz, seq, num_head, d_k]
        
        # Step 2: 转置为 [bsz, num_head, seq, d_k]（重要！）
        q = q.transpose(1, 2)  # [bsz, num_head, seq, d_k]
        k = k.transpose(1, 2)  # [bsz, num_head, seq, d_k]
        v = v.transpose(1, 2)  # [bsz, num_head, seq, d_k]
        
        # Step 3: 计算注意力分数
        # k.transpose(-1, -2): [bsz, num_head, d_k, seq]
        # scores: [bsz, num_head, seq, seq]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        # Step 4: 应用 mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # [bsz, num_head, seq, seq]
        
        # Step 5: Softmax
        scores_soft = F.softmax(scores, dim=-1)  # [bsz, num_head, seq, seq]
        
        # Step 6: 加权求和
        output = torch.matmul(scores_soft, v)  # [bsz, num_head, seq, d_k]
        
        # Step 7: 合并多头
        # transpose: [bsz, seq, num_head, d_k]
        # contiguous: 确保内存连续
        # view: [bsz, seq, d_model]
        output = output.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        
        # Step 8: 最终线性变换
        output = self.W_O(output)  # [bsz, seq, d_model]
        
        return output


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 参数设置
    bsz = 2
    seq = 10
    d_model = 512
    num_head = 8
    
    # 创建模型和数据
    mha = MultiHeadAttention(num_head, d_model)
    x = torch.randn(bsz, seq, d_model)
    
    # 创建 mask（例如：padding mask）
    mask = torch.ones(bsz, 1, 1, seq)  # [bsz, 1, 1, seq]
    mask[:, :, :, 5:] = 0  # 假设后 5 个位置被 mask
    
    # 前向传播
    output = mha(x, mask)
    
    print(f"输入 x shape: {x.shape}")
    print(f"输出 output shape: {output.shape}")
    print(f"期望: [{bsz}, {seq}, {d_model}]")
    
    # 详细的 shape 追踪
    print("\n========== Shape 变化追踪 ==========")
    with torch.no_grad():
        q = mha.W_Q(x).reshape(bsz, seq, num_head, d_model // num_head)
        print(f"1. Q 初始 reshape: {q.shape}")
        
        q = q.transpose(1, 2)
        print(f"2. Q transpose 后: {q.shape}")
        
        k = mha.W_K(x).reshape(bsz, seq, num_head, d_model // num_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2))
        print(f"3. Scores shape: {scores.shape}")
        
        v = mha.W_V(x).reshape(bsz, seq, num_head, d_model // num_head).transpose(1, 2)
        attn_output = torch.matmul(F.softmax(scores, dim=-1), v)
        print(f"4. Attention output: {attn_output.shape}")
        
        attn_output = attn_output.transpose(1, 2)
        print(f"5. Transpose 回来: {attn_output.shape}")
        
        attn_output = attn_output.contiguous().view(bsz, seq, d_model)
        print(f"6. 最终 view: {attn_output.shape}")
