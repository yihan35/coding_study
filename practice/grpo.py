import torch

class GRPO:
    def __init__(self,eps,clip,beta):
        self.eps = eps
        self.clip = clip

    def adv(self,rewards):
        mean = torch.mean(rewards)
        std = torch.std(rewards,unbiased = False)
        advantages = (rewards - mean) / (std+ self.eps)
        return advantages.datach()

    def grpo_loss(self,old_logprobs,new_logprobs,ref_logprobs,advantages,adv_mode):
        ratio = torch.exp(new_logprobs - old_logprobs)
        ref_logr = new_logprobs - ref_logprobs
        kl_score = (torch.exp(ref_logr) - ref_logr -1) * self.beta
        surr1 = ratio * ( advantages + adv_mode)
        surr2 = torch.clamp(ratio,1-self.clip,1+self.clip) * ( advantages + adv_mode)
        loss = -torch.mean(torch.min(surr1,surr2) - kl_score)
        return loss