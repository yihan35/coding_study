import torch
import torch.nn as nn

class SFTCrossEntropyLoss(nn.Module):
    def __init__(self, d_model, vocab_size, ignore_index=-100):
        super().__init__()
        # 1. 语言模型头 (LM Head)：用于将隐状态投影到词表维度
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 2. 交叉熵损失：ignore_index 用于忽略 padding token 的 loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.vocab_size = vocab_size

    def forward(self, hidden_states, labels):
        """
        Args:
            hidden_states: 模型的输出隐状态
                           Shape: [batch_size, seq_len, d_model]
            labels: 真实的标签 (Input IDs)
                    Shape: [batch_size, seq_len]
        """
        
        # --- 步骤 1: 投影到词表空间 ---
        # 输入 Shape: [bsz, seq_len, d_model]
        logits = self.lm_head(hidden_states)
        # 输出 Shape: [bsz, seq_len, vocab_size]
        
        # --- 步骤 2: Shift (错位) 操作 ---
        # SFT 的核心逻辑：第 t 个 token 的输出应该预测第 t+1 个 token 的 label。
        # 因此，我们需要去掉 logits 的最后一位（因为它没有对应的下一位 label），
        # 并去掉 labels 的第一位（因为它是输入，没有对应的 logits 预测它）。
        
        # 取前 seq_len-1 个预测值
        shift_logits = logits[..., :-1, :].contiguous()
        # Shape: [bsz, seq_len - 1, vocab_size]
        
        # 取后 seq_len-1 个真实标签
        shift_labels = labels[..., 1:].contiguous()
        # Shape: [bsz, seq_len - 1]

        # --- 步骤 3: Flatten (展平) ---
        # CrossEntropyLoss 期望输入是 (N, C) 和 (N)
        
        # 将 batch 和 seq 维度合并
        shift_logits = shift_logits.view(-1, self.vocab_size)
        # Shape: [bsz * (seq_len - 1), vocab_size]
        
        shift_labels = shift_labels.view(-1)
        # Shape: [bsz * (seq_len - 1)]

        # --- 步骤 4: 计算 Loss ---
        loss = self.loss_fct(shift_logits, shift_labels)
        # Shape: scalar (标量)
        
        return loss

# --- 使用示例 ---
# 假设参数
bsz = 2
seq_len = 10
d_model = 768
vocab_size = 30000

# 模拟输入
hidden_states = torch.randn(bsz, seq_len, d_model) # [2, 10, 768]
labels = torch.randint(0, vocab_size, (bsz, seq_len)) # [2, 10]

# 初始化并计算
loss_module = SFTCrossEntropyLoss(d_model, vocab_size)
loss = loss_module(hidden_states, labels)

print(f"Loss value: {loss.item()}")