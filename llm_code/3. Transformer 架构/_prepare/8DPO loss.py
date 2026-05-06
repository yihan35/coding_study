import torch
from torch import nn
from torch.nn import functional as F

class DPO:
    def __init__(self, beta):
        self.beta = beta

    def dpo_loss(self, policy_chosen_logps, policy_reject_logps, ref_chosen_logps, ref_reject_logps):
        chosen_r = policy_chosen_logps - ref_chosen_logps
        reject_r = policy_reject_logps - ref_reject_logps
        loss = -F.logsigmoid(self.beta * (chosen_r - reject_r))
        return loss.mean()