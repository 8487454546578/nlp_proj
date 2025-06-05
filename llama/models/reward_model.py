import torch
import torch.nn as nn

# class RewardModel(nn.Module):
#     def __init__(self, base_model, hidden_dim=512):
#         super().__init__()
#         self.base = base_model
#         self.reward_head = nn.Linear(hidden_dim, 1)
#         for param in self.base.parameters():
#             param.requires_grad = False

#     def forward(self, input_ids):
#         tok_emb = self.base.token_embedding(input_ids)
#         x = tok_emb + self.base.pos_embedding[:, :input_ids.size(1), :]
#         x = self.base.blocks(x)
#         x = self.base.norm(x)
#         last_hidden = x[:, -1, :]
#         reward = self.reward_head(last_hidden)
#         return reward.squeeze(-1)

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim=1024):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)

        # 冻结 base model 的参数（可选）
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # 获取 embedding 和位置编码之和后的输出
        x = self.base.token_embedding(input_ids)  # [B, T, D]
        for block in self.base.blocks:
            x = block(x, attention_mask)

        x = self.base.norm(x)  # [B, T, D]

        # 使用最后一个 token 的 hidden state 作为 reward 输入
        last_hidden = x[:, -1, :]  # [B, D]
        reward = self.reward_head(last_hidden)  # [B, 1]

        return reward.squeeze(-1)  # [B]
