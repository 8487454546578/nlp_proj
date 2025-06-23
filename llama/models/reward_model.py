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

# class RewardModel(nn.Module):
#     def __init__(self, base_model, hidden_dim=1024):
#         super().__init__()
#         self.base = base_model
#         self.reward_head = nn.Linear(hidden_dim, 1)

#         # 冻结 base model 的参数（可选）
#         for param in self.base.parameters():
#             param.requires_grad = False

#     def forward(self, input_ids, attention_mask=None):
#         # 获取 embedding 和位置编码之和后的输出
#         x = self.base.token_embedding(input_ids)  # [B, T, D]
#         for block in self.base.blocks:
#             x = block(x, attention_mask)

#         x = self.base.norm(x)  # [B, T, D]

#         # 使用最后一个 token 的 hidden state 作为 reward 输入
#         last_hidden = x[:, -1, :]  # [B, D]
#         reward = self.reward_head(last_hidden)  # [B, 1]

#         return reward.squeeze(-1)  # [B]

# class RewardModel(nn.Module):
#     def __init__(self, base_model, hidden_dim=1024, freeze_base=True):
#         super().__init__()
#         self.base = base_model
#         self.reward_head = nn.Linear(hidden_dim, 1)

#         if freeze_base:
#             for param in self.base.parameters():
#                 param.requires_grad = False

#             # 解冻最后两层 block
#             num_blocks = len(self.base.blocks)
#             for i in range(num_blocks - 1, num_blocks):
#                 for param in self.base.blocks[i].parameters():
#                     param.requires_grad = True

#             print(f"✅ 解冻 base_model 最后 {1} 层（共 {num_blocks} 层）以进行微调")

#     def forward(self, input_ids, attention_mask=None):
#         x = self.base.token_embedding(input_ids)
#         for block in self.base.blocks:
#             x = block(x, attention_mask)
#         x = self.base.norm(x)

#         if attention_mask is not None:
#             attention_mask = attention_mask.unsqueeze(-1)
#             x = x * attention_mask
#             pooled = x.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-6)
#         else:
#             pooled = x.mean(dim=1)

#         reward = self.reward_head(pooled)
#         return reward.squeeze(-1)



class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim=1024, freeze_base=True):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # 1. 获取 token embedding
        x = self.base.token_embedding(input_ids)  # shape: (B, T, D)
        # 2. 经过 transformer blocks
        for block in self.base.blocks:
            x = block(x, attention_mask)
        # 3. LayerNorm
        x = self.base.norm(x)
        # 4. Apply attention mask for pooled representation
        if attention_mask is not None:
            # 将 attention_mask 扩展到与 x 相同的维度 (B, T, 1)
            extended_mask = attention_mask.unsqueeze(-1).float()
            x = x * extended_mask
            pooled = x.sum(dim=1) / extended_mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = x.mean(dim=1)
        # 5. MLP + Sigmoid 输出 reward 值 ∈ [0, 1]
        reward = self.reward_head(pooled)  # shape: (B, 1)
        return reward.squeeze(-1)          # shape: (B,)