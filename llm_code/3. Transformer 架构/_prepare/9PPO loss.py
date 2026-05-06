import torch
from torch import nn
from torch.nn import functional as F

class PPO:
    def __init__(self, clip=0.2, gamma=1, lam=0.95):
        self.clip = clip
        self.gamma = gamma
        self.lam = lam

    def mask_mean(self, loss, mask, dim=-1):
        return (loss * mask).sum(dim=dim) / mask.sum(dim=dim)

    def advantage_estimate(self, rewards, values):
        ''' GAE 广义优势估计 '''
        seq_len = values.shape[1]
        advantages = torch.zeros_like(rewards)
        gae = 0
        for i in range(seq_len-1, -1, -1):
            next_value = values[:, i+1] if i < seq_len-1 else 0.0
            delta = rewards[:, i] + self.gamma * next_value - values[:, i]
            gae = delta + self.lam * self.gamma * gae
            advantages[:, i] = gae
        returns = advantages + values
        return advantages, returns

    def policy_loss(self, new_probs, old_probs, advantages, act_mask):
        ratio = torch.exp(new_probs - old_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantages
        loss = -torch.min(surr1, surr2)
        return self.mask_mean(loss, act_mask)

    def value_loss(self, new_values, returns, act_mask):
        loss = (new_values - returns) ** 2
        return self.mask_mean(loss, act_mask)