import torch
import torch.nn.functional as F

class DPO:
    def __init__(self, beta=0.1, eps=1e-8) -> None:
        self.beta = beta
        self.eps = eps

    def dpo_loss(self, policy_chosen_logps, policy_reject_logps, ref_chosen_logps, ref_reject_logps):
        # policy_chosen_logps, policy_reject_logps: [B]
        # ref_chosen_logps, ref_reject_logps: [B]
        
        # 优答：策略 - 参考
        chosen_logr = policy_chosen_logps - ref_chosen_logps
        # 劣答：策略 - 参考
        reject_logr = policy_reject_logps - ref_reject_logps
        
        # DPO 核心目标
        logits = self.beta * (chosen_logr - reject_logr)
        loss = -F.logsigmoid(logits)
        
        return loss.mean()




        