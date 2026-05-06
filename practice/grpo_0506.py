import torch
class GRPO:
    def __init__(self,eps,clip,beta):
        self.eps = eps
        self.clip = clip
        self.beta = beta
    
    def adv(self,rewards):
        # rewards [B ,G],adv [B,G]
        mean = rewards.mean(dim=-1,keepdim = True)
        std = rewards.std(dim=-1,keepdim = True,unbiased=False)
        adv = (rewards - mean) / (std + self.eps)
        return adv.detach()

    def mask_mean(self,x,mask,dim=-1):
        sum_x = (x*mask).sum(dim =dim)
        sum_mask = mask.sum(dim = dim)
        return sum_x / (sum_mask+self.eps)
    
    def loss(self,old_logps,new_logps,ref_logps,adv,act_mask):
        # old_logps [bsz,group,seq]
        ratio = torch.exp(new_logps - old_logps)
        log_ratio_ref = torch.exp(new_logps-ref_logps)
        kl_penalty =self.beta( torch.exp(log_ratio_ref) - log_ratio_ref -1)
        adv = adv.unsqueeze(-1).expand_as(old_logps)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio,1-self.clip,1+self.clip)* adv
        policy_obj = torch.min(surr1,surr2) - kl_penalty
        # 长度归一化
        seq_obj = self.mask_mean(policy_obj,act_mask,dim=-1)
        # 组内归一化
        group_obj = policy_obj.means(dim =-1)
        loss = -group_obj.mean()
        return loss
        
