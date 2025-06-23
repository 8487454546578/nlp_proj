# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.nn import BCEWithLogitsLoss
# from torch.optim import Adam
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from tokenizers import Tokenizer
# import sys
# sys.path.append(os.path.abspath(".."))
# from llama.models.transformer import TinyTransformer
# from llama.models.reward_model import RewardModel


# # ==== 超参数与设备 ====
# BATCH_SIZE = 16
# EPOCHS = 5
# LR = 1e-4
# MAX_LEN = 512

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==== 加载 tokenizer 和 base model ====
# tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
# vocab_size = tokenizer.get_vocab_size()

# base_model = TinyTransformer(vocab_size=vocab_size)
# base_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
# base_model.to(device)
# base_model.eval()

# # ==== 定义奖励模型 ====
# reward_model = RewardModel(base_model).to(device)

# class RewardDataset(Dataset):
#     def __init__(self, path,max_len=1024):
#         self.samples = []
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 d = json.loads(line)

#                 # 优先使用 safer_response_id，其次 fallback 到 better_response_id
#                 label_id = d.get("safer_response_id", d.get("better_response_id", None))
#                 if label_id not in [0, 1]:
#                     continue  # 如果两者都没有，就跳过该样本

#                 for i, r in enumerate([d["response_0"], d["response_1"]]):
#                     text = d["prompt"] + r
#                     enc = tokenizer.encode(text)
#                     input_ids = enc.ids[:max_len]
#                     label = 1.0 if i == label_id else 0.0
#                     self.samples.append((input_ids, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         input_ids, label = self.samples[idx]
#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         label = torch.tensor(label, dtype=torch.float)
#         return input_ids, label


# def collate_fn(batch):
#     input_ids, labels = zip(*batch)
#     input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
#     return input_ids.to(device), torch.tensor(labels, dtype=torch.float).to(device)

# # ==== 数据加载 ====
# train_dataset = RewardDataset("./data/train.jsonl")
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# # ==== 优化器与损失函数 ====
# optimizer = Adam(reward_model.reward_head.parameters(), lr=LR)
# criterion = BCEWithLogitsLoss()

# # ==== 训练循环 ====
# loss_list = []
# reward_mean_list = []
# reward_model.train()

# for epoch in range(EPOCHS):
#     total_loss = 0
#     total_reward = 0
#     count = 0

#     for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#         rewards = reward_model(input_ids)
#         loss = criterion(rewards, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_reward += rewards.detach().mean().item()
#         count += 1

#     avg_loss = total_loss / count
#     avg_reward = total_reward / count
#     loss_list.append(avg_loss)
#     reward_mean_list.append(avg_reward)

#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

# # ==== 保存结果 ====
# os.makedirs("rlhf_pic", exist_ok=True)
# plt.figure(figsize=(10, 5))
# plt.plot(loss_list, label="Loss")
# plt.plot(reward_mean_list, label="Avg Reward")
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.title("Reward Model Training")
# plt.legend()
# plt.grid(True)
# plt.savefig("rlhf_pic/loss_reward.png")
# plt.close()

# # 保存模型
# torch.save(reward_model.state_dict(), "reward_model.pt")

# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, Subset
# from torch.nn import BCEWithLogitsLoss
# from torch.optim import Adam
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from tokenizers import Tokenizer
# from sklearn.model_selection import train_test_split
# import sys

# # ==== 模型路径设置 ====
# sys.path.append(os.path.abspath(".."))
# from llama.models.transformer import TinyTransformer
# from llama.models.reward_model import RewardModel

# # ==== 超参数与设备 ====
# BATCH_SIZE = 16
# EPOCHS = 5
# LR = 1e-4
# MAX_LEN = 512

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==== 加载 tokenizer 和 base model ====
# tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
# vocab_size = tokenizer.get_vocab_size()

# base_model = TinyTransformer(vocab_size=vocab_size)
# base_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
# base_model.to(device)

# # ==== 定义奖励模型 ====
# reward_model = RewardModel(base_model).to(device)

# # ==== 数据集定义 ====
# class RewardDataset(Dataset):
#     def __init__(self, path, max_len=1024):
#         self.samples = []
#         self.tokenizer = tokenizer  # 假设已定义
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 d = json.loads(line)

#                 # 选出更安全的回答 ID
#                 safer_id = d.get("safer_response_id", d.get("better_response_id", None))
#                 if safer_id not in [0, 1]:
#                     continue

#                 # 定义 safer 和 less safe 的回答
#                 r = d[f"response_{safer_id}"]
#                 l = d[f"response_{1 - safer_id}"]

#                 # 处理 safer → label 1
#                 enc_r = self.tokenizer.encode(d["prompt"] + r)
#                 input_ids_r = enc_r.ids[:max_len]
#                 self.samples.append((input_ids_r, 1.0))

#                 # 处理 less safe → label 0
#                 enc_l = self.tokenizer.encode(d["prompt"] + l)
#                 input_ids_l = enc_l.ids[:max_len]
#                 self.samples.append((input_ids_l, 0.0))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         input_ids, label = self.samples[idx]
#         input_ids = torch.tensor(input_ids, dtype=torch.long)
#         label = torch.tensor(label, dtype=torch.float)
#         return input_ids, label


# # ==== collate 函数 ====
# def collate_fn(batch):
#     input_ids, labels = zip(*batch)
#     input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
#     return input_ids.to(device), torch.tensor(labels, dtype=torch.float).to(device)

# # ==== 加载数据集并划分训练/验证集 ====
# full_dataset = RewardDataset("./data/train.jsonl")
# train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, random_state=42)
# train_dataset = Subset(full_dataset, train_idx)
# val_dataset = Subset(full_dataset, val_idx)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# # ==== 验证函数 ====
# @torch.no_grad()
# def evaluate_accuracy(model, val_loader):
#     model.eval()
#     total_correct = 0
#     total = 0
#     for input_ids, labels in val_loader:
#         logits = model(input_ids)
#         preds = (torch.sigmoid(logits) > 0.5).float()
#         total_correct += (preds == labels).sum().item()
#         total += labels.numel()
#     model.train()
#     return total_correct / total if total > 0 else 0

# # ==== 优化器与损失函数 ====
# optimizer = Adam(reward_model.reward_head.parameters(), lr=LR)
# criterion = BCEWithLogitsLoss()

# # ==== 训练循环 ====
# loss_list = []
# reward_mean_list = []
# val_acc_list = []
# best_val_acc = 0.0

# reward_model.train()

# for epoch in range(EPOCHS):
#     total_loss = 0
#     total_reward = 0
#     count = 0

#     for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#         rewards = reward_model(input_ids)
#         loss = criterion(rewards, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_reward += rewards.detach().mean().item()
#         count += 1

#     avg_loss = total_loss / count
#     avg_reward = total_reward / count
#     loss_list.append(avg_loss)
#     reward_mean_list.append(avg_reward)

#     # === 评估验证集准确率 ===
#     val_acc = evaluate_accuracy(reward_model, val_loader)
#     val_acc_list.append(val_acc)
#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

#     # === 保存最优模型 ===
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(reward_model.state_dict(), "reward_model_best.pt")
#         print(f"✅ Saved best model at epoch {epoch+1} with acc {val_acc:.4f}")

# # ==== 可视化 ====
# os.makedirs("rlhf_pic", exist_ok=True)
# plt.figure(figsize=(10, 5))
# plt.plot(loss_list, label="Loss")
# plt.plot(val_acc_list, label="Val Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.title("Reward Model Training")
# plt.legend()
# plt.grid(True)
# plt.savefig("rlhf_pic/loss_acc.png")
# plt.close()



# import os
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, Subset
# from torch.optim import Adam
# from tqdm import tqdm
# from tokenizers import Tokenizer
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # ==== 设置超参数与设备 ====
# BATCH_SIZE = 16
# EPOCHS = 5
# LR = 1e-4
# MAX_LEN = 512

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ==== 加载 tokenizer 和 base model ====
# tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
# vocab_size = tokenizer.get_vocab_size()

# # ==== 定义 TinyTransformer 和 RewardModel ====
# from llama.models.transformer import TinyTransformer
# from llama.models.reward_model import RewardModel

# base_model = TinyTransformer(vocab_size=vocab_size)
# base_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
# base_model.to(device)
# reward_model = RewardModel(base_model).to(device)

# # ==== Pairwise Dataset ====
# class PairwiseRewardDataset(Dataset):
#     def __init__(self, path, max_len=512):
#         self.samples = []
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 d = json.loads(line)
#                 better_id = d.get("better_response_id")
#                 if better_id not in [0, 1]:
#                     continue
#                 worse_id = 1 - better_id
#                 r_better = d[f"response_{better_id}"]
#                 r_worse = d[f"response_{worse_id}"]
#                 prompt = d["prompt"]

#                 ids_b = tokenizer.encode(prompt + r_better).ids[:max_len]
#                 ids_w = tokenizer.encode(prompt + r_worse).ids[:max_len]
#                 self.samples.append((ids_b, ids_w))

#     def __len__(self): return len(self.samples)

#     def __getitem__(self, idx):
#         better, worse = self.samples[idx]
#         return torch.tensor(better), torch.tensor(worse)

# # ==== collate 函数 ====
# def collate_pairwise_fn(batch):
#     better, worse = zip(*batch)
#     better = nn.utils.rnn.pad_sequence(better, batch_first=True, padding_value=0)
#     worse = nn.utils.rnn.pad_sequence(worse, batch_first=True, padding_value=0)
#     return better.to(device), worse.to(device)

# # ==== 加载数据 ====
# full_dataset = PairwiseRewardDataset("./data/train.jsonl")
# train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, random_state=42)
# train_dataset = Subset(full_dataset, train_idx)
# val_dataset = Subset(full_dataset, val_idx)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairwise_fn)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pairwise_fn)

# # ==== 评估准确率 ====
# @torch.no_grad()
# def evaluate_pairwise_accuracy(model, val_loader):
#     model.eval()
#     correct, total = 0, 0
#     for better_ids, worse_ids in val_loader:
#         r_b = model(better_ids)
#         r_w = model(worse_ids)
#         correct += (r_b > r_w).sum().item()
#         total += r_b.size(0)
#     model.train()
#     return correct / total

# # ==== 训练 ====
# optimizer = Adam(reward_model.parameters(), lr=LR)
# loss_list = []
# acc_list = []
# best_acc = 0

# for epoch in range(EPOCHS):
#     reward_model.train()
#     total_loss = 0
#     for better_ids, worse_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#         r_b = reward_model(better_ids)
#         r_w = reward_model(worse_ids)
#         target = torch.ones_like(r_b)
#         loss = nn.functional.margin_ranking_loss(r_b, r_w, target, margin=0.5)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)
#     acc = evaluate_pairwise_accuracy(reward_model, val_loader)
#     loss_list.append(avg_loss)
#     acc_list.append(acc)

#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
#     if acc > best_acc:
#         best_acc = acc
#         torch.save(reward_model.state_dict(), "reward_model_pairwise_best.pt")
#         print(f"✅ Saved best model at epoch {epoch+1} with acc {acc:.4f}")

# # ==== 可视化 ====
# os.makedirs("rlhf_pic", exist_ok=True)
# plt.figure(figsize=(10, 5))
# plt.plot(loss_list, label="Train Loss")
# plt.plot(acc_list, label="Val Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Value")
# plt.title("Pairwise Reward Model Training")
# plt.legend()
# plt.grid(True)
# plt.savefig("rlhf_pic/pairwise_loss_acc.png")
# plt.close()


import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from llama.utils.seed import set_seed
# ==== 设置超参数与设备 ====
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
MAX_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载 tokenizer 和 base model ====
tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

from llama.models.transformer import TinyTransformer
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

            num_blocks = len(self.base.blocks)
            for i in range(num_blocks - 1, num_blocks):
                for param in self.base.blocks[i].parameters():
                    param.requires_grad = True

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


# ==== 初始化模型 ====
base_model = TinyTransformer(vocab_size=vocab_size)
base_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
base_model.to(device)
reward_model = RewardModel(base_model, freeze_base=True).to(device)

# ==== Pairwise Dataset ====
class PairwiseRewardDataset(Dataset):
    def __init__(self, path, max_len=1024):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                better_id = d.get("safer_response_id")
                if better_id not in [0, 1]:
                    continue
                worse_id = 1 - better_id
                r_better = d[f"response_{better_id}"]
                r_worse = d[f"response_{worse_id}"]
                prompt = d["prompt"]

                ids_b = tokenizer.encode(prompt + r_better).ids[:max_len]
                ids_w = tokenizer.encode(prompt + r_worse).ids[:max_len]
                self.samples.append((ids_b, ids_w))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        better, worse = self.samples[idx]
        return torch.tensor(better), torch.tensor(worse)

def collate_pairwise_fn(batch):
    better, worse = zip(*batch)
    better = nn.utils.rnn.pad_sequence(better, batch_first=True, padding_value=0)
    worse = nn.utils.rnn.pad_sequence(worse, batch_first=True, padding_value=0)
    return better.to(device), worse.to(device)

# ==== 加载数据 ====
full_dataset = PairwiseRewardDataset("./data/train_filter.jsonl")
train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, random_state=42)
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = PairwiseRewardDataset("./data/test_filter.jsonl")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pairwise_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pairwise_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pairwise_fn)

# ==== 评估准确率 ====
@torch.no_grad()
def evaluate_pairwise_accuracy(model, val_loader):
    model.eval()
    correct, total = 0, 0
    for better_ids, worse_ids in val_loader:
        mask_b = (better_ids != 0).long()
        mask_w = (worse_ids != 0).long()
        r_b = model(better_ids, attention_mask=mask_b)
        r_w = model(worse_ids, attention_mask=mask_w)
        correct += (r_b > r_w).sum().item()
        total += r_b.size(0)
    model.train()
    return correct / total

# ==== 训练 ====
trainable_params = filter(lambda p: p.requires_grad, reward_model.parameters())
optimizer = Adam(trainable_params, lr=LR)

loss_list = []
acc_list = []
best_acc = 0.0

for epoch in range(EPOCHS):
    set_seed(42)
    reward_model.train()
    total_loss = 0
    r_b_all, r_w_all = [], []

    for better_ids, worse_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        mask_b = (better_ids != 0).long()
        mask_w = (worse_ids != 0).long()

        r_b = reward_model(better_ids, attention_mask=mask_b)
        r_w = reward_model(worse_ids, attention_mask=mask_w)
        r_b_all += r_b.detach().cpu().tolist()
        r_w_all += r_w.detach().cpu().tolist()

        # Pairwise loss
        loss = -F.logsigmoid(r_b - r_w).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc = evaluate_pairwise_accuracy(reward_model, val_loader)
    loss_list.append(avg_loss)
    acc_list.append(acc)

    print(f"\nEpoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
  
    if acc > best_acc:
        best_acc = acc
        torch.save(reward_model.state_dict(), "reward_model_pairwise_best_for_eval.pt")
        print(f"Saved best model at epoch {epoch+1} with acc {acc:.4f}")

# ==== 加载最佳模型并评估测试集准确率 ====
print("\nEvaluating best model on test set...")
best_model = RewardModel(base_model, freeze_base=True).to(device)
best_model.load_state_dict(torch.load("reward_model_pairwise_best_for_eval.pt"))
test_acc = evaluate_pairwise_accuracy(best_model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")


# ==== 可视化 ====
os.makedirs("rlhf_pic", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(loss_list, label="Train Loss")
plt.plot(acc_list, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Pairwise Reward Model for evaluate")
plt.legend()
plt.grid(True)
plt.savefig("rlhf_pic/pairwise_loss_acc_for_eval.png")
plt.close()
