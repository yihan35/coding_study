import torch

class GRPO:
    def __init__(self, eps, clip, beta):
        self.eps = eps
        self.clip = clip
        self.beta = beta

    def group_advantage(self, rewards):
        mean = torch.mean(rewards)
        std = torch.std(rewards, unbiased=False) # 有偏，总体标准差
        advantages = (rewards - mean) / (std + self.eps)
        return advantages.detach()

    def grpo_loss(self, old_logps, new_logps, ref_logps, advantages):
        ratio = torch.exp(new_logps - old_logps)
        ref_logr = new_logps - ref_logps
        kl_score = (torch.exp(ref_logr) - ref_logr - 1) * self.beta
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantages
        loss = -torch.mean(torch.min(surr1, surr2) - kl_score)
        return loss

    def mask_mean(self, loss, mask, dim=-1):
        return (loss * mask).sum(dim=dim) / mask.sum(dim=dim)