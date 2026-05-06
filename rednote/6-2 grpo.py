import torch

class GRPO:
    def __init__(self,eps = 1e-8,clip = 0.2,beta = 0.01) -> None:
        self.eps = eps
        self.clip = clip
        self.beta = beta

    def group_advantage(self, rewards):
        # rewards shape: (B, G) —— 组维度是 G，在 dim=1 求均值
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean) / (std + self.eps)
        return advantages.detach()

    def mask_mean(self, x, mask, dim = -1):
        """带掩码的 token 平均 (B, G, S) → (B, G)"""
        # x ，mask：shape：[B,G,S]; 输出：[B,G]
        sum_x = (x * mask).sum(dim=dim)      # 有效位置的 r*A 相加
        sum_mask = mask.sum(dim=dim)         # 有效位置的长度
        return sum_x / (sum_mask + self.eps) # 长度归一化

    def grpo_loss(self, old_logps, new_logps, ref_logps, advantages, act_mask):
        # old_logps，new_logps，ref_logps，act_mask：[B,G,S];
        # advantages：[B,G]
        # ==================== 1. 重要性采样比率 ====================
        ratio = torch.exp(new_logps - old_logps)  # (B, G, S)

        # ==================== 2. KL 散度惩罚 ====================
        log_ratio_ref = new_logps - ref_logps
        kl_penalty = self.beta * (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0)

        # ==================== 3. 优势广播到 token 维度 ====================
        # (B, G) → (B, G, 1) → (B, G, S)
        adv = advantages.unsqueeze(-1).expand_as(old_logps)

        # ==================== 4. GRPO 裁剪目标 ====================
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
        policy_obj = torch.min(surr1, surr2) - kl_penalty  # (B, G, S)

        # ==================== 5. 第一层平均：Token 归一化 1/|o_i| ====================
        seq_obj = self.mask_mean(policy_obj, act_mask, dim=-1)  # (B, G)

        # ==================== 6. 第二层平均：组归一化 1/G ====================
        group_obj = seq_obj.mean(dim=1)  # (B,) —— 1/G sum_{i=1}^G

        # ==================== 7. 第三层：批次期望（最终损失） ====================
        loss = -group_obj.mean()  # 梯度上升转下降

        return loss