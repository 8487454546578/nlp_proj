import os
import sys
import json
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(".."))
from llama.models.transformer import TinyTransformer
from llama.models.reward_model import RewardModel
from llama.datasets.pku_safe import PKUSafeDataset
from llama.utils.evaluate_safety import evaluate_safety_rate
from llama.utils.generation import generate_response
from llama.utils.loss import ppo_loss
from llama.utils.reward_utils import compute_reward

# ==== 设置设备 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载 tokenizer 和模型 ====
tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

policy_model = TinyTransformer(vocab_size=vocab_size).to(device)
policy_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
policy_model.train()

base_model = TinyTransformer(vocab_size=vocab_size).to(device)
reward_model = RewardModel(base_model).to(device)
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()

# ==== 加载数据 ====
train_dataset = PKUSafeDataset("./data/train.jsonl")
test_dataset = PKUSafeDataset("./data/test.jsonl")
subset_dataset = torch.utils.data.Subset(train_dataset, list(range(min(1000, len(train_dataset)))))
test_subset = [test_dataset[i] for i in range(min(100, len(test_dataset)))]

train_loader = DataLoader(subset_dataset, batch_size=8, shuffle=True)
print("训练集样本数量:", len(subset_dataset))
print("测试子集样本数量:", len(test_subset))

# ==== 定义 LoRA 结构 ====
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r=8, lora_alpha=32):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.bias = orig_linear.bias is not None

        self.weight = orig_linear.weight
        self.bias = orig_linear.bias

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.out_features, r) * 0.01)

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        lora_update = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return result + lora_update

    def to(self, *args, **kwargs):
        self.lora_A = nn.Parameter(self.lora_A.to(*args, **kwargs))
        self.lora_B = nn.Parameter(self.lora_B.to(*args, **kwargs))
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return super().to(*args, **kwargs)

def replace_linear_with_lora(model, target_modules, r=8, lora_alpha=32):
    for name in target_modules:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        last = parts[-1]
        orig_layer = getattr(parent, last)
        if isinstance(orig_layer, nn.Linear):
            setattr(parent, last, LoRALinear(orig_layer, r, lora_alpha))
            print(f"[LoRA] Replaced {name} with LoRALinear.")
        else:
            print(f"[LoRA][Skip] {name} is not nn.Linear.")
    return model

# 替换所有指定模块为 LoRA
target_modules = []
for i in range(16):  # 修改为你的模型层数
    target_modules += [
        f"blocks.{i}.attn.out_proj",
        f"blocks.{i}.ffn.w1",
        f"blocks.{i}.ffn.w2",
        f"blocks.{i}.ffn.w3"
    ]
policy_model = replace_linear_with_lora(policy_model, target_modules, r=8, lora_alpha=32).to(device)

# 冻结原参数，仅训练 LoRA
for name, param in policy_model.named_parameters():
    param.requires_grad = "lora_" in name

# PPO 优化器
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=1e-5)

# ==== PPO 训练循环 ====
log_file = open("ppo_training_log.txt", "a")
reward_curve, loss_curve, safety_curve = [], [], []

for epoch in range(20):
    total_loss, total_reward = 0.0, 0.0

    for prompts in tqdm(train_loader, desc=f"Epoch {epoch}"):
        prompt = prompts[0]  # 仅使用一个 prompt（可扩展 batch）
        response = generate_response(policy_model, prompt, tokenizer)
        reward = compute_reward(prompt, response, reward_model=reward_model, tokenizer=tokenizer)
        total_reward += reward

        text = prompt + " " + response
        input_ids = torch.tensor([tokenizer.encode(text).ids], device=device)

        with torch.no_grad():
            logits = policy_model(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            old_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

        advantage = torch.tensor([reward], device=device)
        loss = ppo_loss(policy_model, old_log_probs, input_ids, advantage)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 安全评估与日志记录
    safety_rate = evaluate_safety_rate(policy_model, reward_model, tokenizer, test_dataset=test_subset)
    avg_reward = total_reward / len(train_loader)
    avg_loss = total_loss / len(train_loader)

    reward_curve.append(avg_reward)
    loss_curve.append(avg_loss)
    safety_curve.append(safety_rate)

    log_msg = f"[{datetime.now()}] Epoch {epoch} | Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.4f} | Safety Rate: {safety_rate:.2%}"
    print(log_msg)
    log_file.write(log_msg + "\n")

    if epoch % 5 == 0 or safety_rate > 0.8:
        torch.save(policy_model.state_dict(), f"tiny_model_ppo_epoch{epoch}.pt")

log_file.close()

# ==== 绘制训练曲线 ====
os.makedirs("rlhf_pic", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(reward_curve, label="Avg Reward")
plt.plot(loss_curve, label="Loss")
plt.plot(safety_curve, label="Safety Rate")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("PPO + LoRA Training Curve")
plt.legend()
plt.grid(True)
plt.savefig("rlhf_pic/ppo_lora_curve.png")
plt.close()
