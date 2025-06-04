import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim=512):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        tok_emb = self.base.token_embedding(input_ids)
        x = tok_emb + self.base.pos_embedding[:, :input_ids.size(1), :]
        x = self.base.blocks(x)
        x = self.base.norm(x)
        last_hidden = x[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
