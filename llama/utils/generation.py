import torch
from torch.nn import functional as F

def generate_response(model, prompt, tokenizer, max_new_tokens=64, device="cuda"):
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break
    return tokenizer.decode(input_ids[0].tolist())
