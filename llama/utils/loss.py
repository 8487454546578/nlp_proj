import torch
import torch.nn.functional as F

def ppo_loss(policy_model, old_log_probs, input_ids, advantages):
    logits = policy_model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(selected_log_probs - old_log_probs)
    clip_range = 0.2
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss

