import torch

def compute_reward(prompt, response, reward_model, tokenizer, device="cuda"):
    text = prompt + response
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids], device=device)
    with torch.no_grad():
        reward = reward_model(input_ids)
        return reward.item()
