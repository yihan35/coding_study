import torch

class PPO:
    def __init__(self, clip=0.2, gamma=1.0, lam=0.95, eps=1e-8) -> None:
        self.clip = clip
        self.gamma = gamma # 折扣系数
        self.lam = lam # 优势平滑系数
        self.eps = eps

    def compute_rewards(self, old_logps, ref_logps, scores, act_mask):
        # old_logps, ref_logps, act_mask: [B, S]
        # scores: [B]
        B, S = old_logps.shape
        # 每个 token 的 KL 基础奖励：ref_logp - old_logp
        kl_reward = ref_logps - old_logps
        token_rewards = kl_reward.clone()       
        # 找到每条样本最后一个有效 token 位置
        seq_len = act_mask.sum(dim=-1) - 1  # [B]      
        # 只在最后一个 token 加上 RM 总分
        for i in range(B):
            end_pos = seq_len[i].long()
            token_rewards[i, end_pos] += scores[i]
        return token_rewards  # [B, S]

    def advantage_estimate(self, rewards, values):
        # rewards, values: shape [B, S]
        B, S = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        # 倒着遍历每一步 t
        for t in reversed(range(S)):
            # 下一步的价值
            next_val = values[:, t+1] if t < S-1 else 0.0
            # TD 误差 = 实际收益（当前奖励+折扣未来价值）- 预估收益（当前价值）
            delta = rewards[:, t] + self.gamma * next_val - values[:, t]
            # GAE = 当下好坏（TD误差）+ 未来好坏（把未来优势折回当前步）
            gae = delta + self.lam * self.gamma * gae
            # 保存当前步优势
            advantages[:, t] = gae
        # 真实总回报 = 优势 + 价值
        returns = advantages + values
        return advantages.detach(), returns.detach()

    def mask_mean(self, x, mask, dim=-1):
        """带掩码的 token 平均 [B, S] -> [B]"""
        # x, mask: shape [B, S]; 输出: [B]
        sum_x = (x * mask).sum(dim=dim)
        sum_mask = mask.sum(dim=dim)
        return sum_x / (sum_mask + self.eps)

    def policy_loss(self, new_logps, old_logps, advantages, act_mask):
        # new_logps, old_logps, act_mask: [B, S]
        # advantages: [B, S]
        ratio = torch.exp(new_logps - old_logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages
        policy_obj = torch.min(surr1, surr2)
        seq_loss = self.mask_mean(policy_obj, act_mask, dim=-1)
        loss = -seq_loss.mean()
        return loss

    def value_loss(self, new_values, returns, act_mask):
        # new_values, returns, act_mask: [B, S]
        value_mse = (new_values - returns) ** 2
        seq_loss = self.mask_mean(value_mse, act_mask, dim=-1)
        loss = seq_loss.mean()
        return loss

ppo = PPO()
# 1. 计算逐 token 奖励
rewards = ppo.compute_rewards(old_logps, ref_logps, scores, act_mask)

# 2. 计算优势 & 回报
advantages, returns = ppo.advantage_estimate(rewards, values)

# 3. 计算损失
loss_policy = ppo.policy_loss(new_logps, old_logps, advantages, act_mask)
loss_value = ppo.value_loss(new_values, returns, act_mask)




