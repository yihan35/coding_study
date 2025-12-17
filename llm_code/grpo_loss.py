import torch
def compute_grpo_loss(model,inputs):
    epsilon = 0.2
    prompt_ids ,prompt_mask = inputs["prompt_ids"],inputs["prompt_mask"]
    completion_ids,completion_mask = inputs["completion_ids"],inputs["comlpetion_mask"]
    input_ids = torch.cat([prompt_ids,completion_ids],dim = 1)
    attention_mask = torch.cat([prompt_mask,completion_mask],dim = 1)

    logits_to_keep = completion_ids.size(1)
    per_token_logps = get_per_token_logps(input_ids,attention_mask,logits_to_keep)

    advantages = inputs["advantage"]
    old_per_token_logps = inputs["old_per_token_logps"]

    coef_1 = torch.exp(per_token_logps-old_per_token_logps)
    coef_2 = torch.clamp(coef_1,1-epsilon,1+epsilon)

    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)

    per_token_loss = -torch.min(per_token_loss1,per_token_loss2)

    return per_token_loss.mean()
