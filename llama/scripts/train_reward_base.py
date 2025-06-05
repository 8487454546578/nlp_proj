import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
import sys
sys.path.append(os.path.abspath(".."))
from llama.models.transformer import TinyTransformer
from llama.models.reward_model import RewardModel


# ==== 超参数与设备 ====
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
MAX_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载 tokenizer 和 base model ====
tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

base_model = TinyTransformer(vocab_size=vocab_size)
base_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
base_model.to(device)
base_model.eval()

# ==== 定义奖励模型 ====
reward_model = RewardModel(base_model).to(device)

# ==== 数据集定义 ====
class RewardDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        with open(path, "r") as f:
            for line in f:
                d = json.loads(line)
                for i, r in enumerate([d['response_0'], d['response_1']]):
                    text = d['prompt'] + r
                    enc = tokenizer.encode(text)
                    input_ids = enc.ids[:MAX_LEN]
                    label = 1.0 if d.get('safer_response', 0) == i else 0.0
                    self.samples.append((input_ids, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, label = self.samples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float)
        return input_ids, label

def collate_fn(batch):
    input_ids, labels = zip(*batch)
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    return input_ids.to(device), torch.tensor(labels, dtype=torch.float).to(device)

# ==== 数据加载 ====
train_dataset = RewardDataset("./data/train.jsonl")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==== 优化器与损失函数 ====
optimizer = Adam(reward_model.reward_head.parameters(), lr=LR)
criterion = BCEWithLogitsLoss()

# ==== 训练循环 ====
loss_list = []
reward_mean_list = []
reward_model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    total_reward = 0
    count = 0

    for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        rewards = reward_model(input_ids)
        loss = criterion(rewards, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reward += rewards.detach().mean().item()
        count += 1

    avg_loss = total_loss / count
    avg_reward = total_reward / count
    loss_list.append(avg_loss)
    reward_mean_list.append(avg_reward)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

# ==== 保存结果 ====
os.makedirs("rlhf_pic", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(loss_list, label="Loss")
plt.plot(reward_mean_list, label="Avg Reward")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Reward Model Training")
plt.legend()
plt.grid(True)
plt.savefig("rlhf_pic/loss_reward.png")
plt.close()

# 保存模型
torch.save(reward_model.state_dict(), "reward_model.pt")
