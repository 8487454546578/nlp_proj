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
import gc
sys.path.append(os.path.abspath(".."))
from llama.models.transformer import TinyTransformer
from llama.models.reward_model import RewardModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import random
import json
from torch.utils.data import Dataset

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
reward_model.load_state_dict(torch.load("reward_model_pairwise_best.pt"))
reward_model.eval()
class PKUSafeDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["prompt"]

class PKUSafePairDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "response_0": item["response_0"],
            "response_1": item["response_1"],
            "safer_response_id": item.get("safer_response_id", 1)
        }

    def __len__(self):
        return len(self.data)



# ==== 加载数据 ====
# === 加载 PPO 数据集 (只含 prompt) ===
ppo_dataset = PKUSafeDataset("./data/train.jsonl")
train_loader = DataLoader(torch.utils.data.Subset(ppo_dataset, range(min(1000, len(ppo_dataset)))), batch_size=8, shuffle=True)

# === 加载对比学习数据集 (含 prompt + response_0/1 + label) ===
contrastive_dataset = list(PKUSafePairDataset("./data/train.jsonl"))


import json
import random
from torch.utils.data import Dataset

# ==== 筛选危险样本函数 ====
def is_unsafe(example):
    if not example.get("is_response_0_safe", True) or not example.get("is_response_1_safe", True):
        return True
    if example.get("response_0_severity_level", 0) > 0 or example.get("response_1_severity_level", 0) > 0:
        return True
    if any(example.get("response_0_harm_category", {}).values()):
        return True
    if any(example.get("response_1_harm_category", {}).values()):
        return True
    return False

# ==== 读取 + 筛选100条危险样本 ====
with open("./data/test.jsonl", "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

unsafe_samples = [ex for ex in all_data if is_unsafe(ex)]
selected_unsafe_samples = random.sample(unsafe_samples, min(100, len(unsafe_samples)))

# ==== 构造与原PKUSafeDataset结构一致的 Dataset ====
class UnsafePromptDataset(Dataset):
    def __init__(self, data, with_label=False):
        self.data = data
        self.with_label = with_label

    def __getitem__(self, idx):
        raw = self.data[idx]

        # 保证 prompt 是字符串
        prompt = str(raw.get("prompt", ""))

        # 统一处理两个 response
        response_0 = str(raw.get("response_0", ""))
        response_1 = str(raw.get("response_1", ""))

        # label（可选）
        label = raw.get("better_response_id", None) if self.with_label else None

        return {
            "prompt": prompt,
            "response_0": response_0,
            "response_1": response_1,
            "label": label,
        }

    def __len__(self):
        return len(self.data)

# ==== 替代原 test_dataset 和 test_subset ====
test_dataset = UnsafePromptDataset(unsafe_samples, with_label=True)
test_subset = [test_dataset[i] for i in range(min(100,len(test_dataset)))]
def contrastive_loss(embeddings, labels, temperature=0.07):
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # (B, B)
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B), bool tensor

    # mask out self-similarity
    mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    # 对每个样本计算正样本的相似度之和
    exp_sim = torch.exp(sim_matrix)
    pos_sim = exp_sim * labels.float()

    loss_per_sample = -torch.log(
        pos_sim.sum(dim=1) / (exp_sim.sum(dim=1) + 1e-8) + 1e-8
    )
    loss = loss_per_sample.mean()
    return loss

print("危险样本子集大小:", len(test_subset))


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

# 优化器（只更新可训练参数）
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=1e-5)

# 日志与曲线
log_file = open("ppo_training_log.txt", "a")
reward_curve, loss_curve, safety_curve, contrastive_curve = [], [], [], []

# 对比损失超参
alpha = 0.1  # 对比损失权重
temperature = 0.07  # 对比学习温度参数

# === PPO + Contrastive Training 主循环 ===
for epoch in range(5):
    policy_model.train()
    total_reward, total_loss, total_cl_loss = 0, 0, 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True):
        optimizer.zero_grad()
        batch_ppo_loss = 0

        # === PPO 损失计算 ===
        for prompt in batch:
            response = generate_response(policy_model, prompt, tokenizer)
            reward = compute_reward(prompt, response, reward_model, tokenizer)
            total_reward += reward

            text = prompt + " " + response
            input_ids = torch.tensor([tokenizer.encode(text).ids], device=device)

            with torch.no_grad():
                logits = policy_model(input_ids)
                log_probs = F.log_softmax(logits, dim=-1)
                old_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

            advantage = torch.tensor([reward], device=device)
            loss = ppo_loss(policy_model, old_log_probs, input_ids, advantage)
            batch_ppo_loss += loss

        # === Contrastive 损失计算 ===
        contrastive_batch = random.sample(contrastive_dataset, k=8)
        contrastive_embeddings, contrastive_labels = [], []

        for item in contrastive_batch:
            prompt = item["prompt"]
            for i in [0, 1]:
                response = item[f"response_{i}"]
                label = int(i == item["safer_response_id"])

                text = prompt + " " + response
                input_ids = torch.tensor([tokenizer.encode(text).ids], device=device)
                emb = policy_model(input_ids).mean(dim=1).squeeze(0)
                contrastive_embeddings.append(emb)
                contrastive_labels.append(label)

        # === 合并损失 + 反向传播 ===
        batch_ppo_loss /= len(batch)
        embeddings = torch.stack(contrastive_embeddings)
        labels = torch.tensor(contrastive_labels, device=device)
        cl_loss = contrastive_loss(embeddings, labels, temperature=temperature)
        total_cl_loss += cl_loss.item()

        total_loss_batch = batch_ppo_loss + alpha * cl_loss
        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()

    # === 评估与模型保存 ===
    policy_model.eval()
    safety_rate = evaluate_safety_rate(policy_model, reward_model, tokenizer, test_dataset=test_subset, threshold=0.6)
    safety_rate = evaluate_safety_rate(
        policy_model, reward_model, tokenizer,
        test_dataset=[item["prompt"] for item in test_subset],  # 👈 只传入 prompt
        threshold=0.6
    )
    print(f"[Epoch {epoch}] PPO Loss={total_loss/len(train_loader):.4f} | Contrastive Loss={total_cl_loss/len(train_loader):.4f} | Safety Rate={safety_rate:.2%}")
    torch.save(policy_model.state_dict(), f"tiny_model_ppo_epoch_contrastive{epoch}.pt")
# for epoch in range(20):
#     total_loss_val = 0.0
#     total_reward = 0.0
#     last_loss = None  # 仅保留最后一个 loss 的 graph

#     for prompts in tqdm(train_loader, desc=f"Epoch {epoch}"):
#         prompt = prompts[0]  # 单个样本
#         response = generate_response(policy_model, prompt, tokenizer)
#         reward = compute_reward(prompt, response, reward_model=reward_model, tokenizer=tokenizer)
#         total_reward += reward

#         text = prompt + " " + response
#         input_ids = torch.tensor([tokenizer.encode(text).ids], device=device)

#         # 获取旧策略的 log_probs（PPO 需要）
#         with torch.no_grad():
#             logits = policy_model(input_ids)
#             log_probs = F.log_softmax(logits, dim=-1)
#             old_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

#         advantage = torch.tensor([reward], device=device)
#         loss = ppo_loss(policy_model, old_log_probs, input_ids, advantage)

#         total_loss_val += loss.item()
#         last_loss = loss  # 只保留最后一个有 graph 的 loss

#     # ===== 对比学习阶段（安全性增强） =====
#     policy_model.eval()
#     embedding_list, label_list = [], []

#     for example in test_subset:
#         prompt_text = example["prompt"]
#         label = example["label"]
#         input_ids = torch.tensor([tokenizer.encode(prompt_text).ids], device=device)

#         with torch.no_grad():
#             hidden_states = policy_model(input_ids)  # [1, L, D]
#         prompt_embedding = hidden_states[:, 0, :]  # 取 [CLS] 或 mean pooling
#         embedding_list.append(prompt_embedding.squeeze(0))
#         label_list.append(label)

#     embeddings = torch.stack(embedding_list)  # [N, D]
#     labels = torch.tensor(label_list, dtype=torch.long, device=device)

#     cl_loss = contrastive_loss(embeddings, labels, temperature=temperature)
#     contrastive_curve.append(cl_loss.item())

#     # ========== 联合优化 & 清理图 ==========
#     avg_reward = total_reward / len(train_loader)
#     safety_rate = evaluate_safety_rate(policy_model, reward_model, tokenizer, test_dataset=test_subset)

#     # 使用最后一个 PPO loss 和 contrastive loss 联合优化
#     total_epoch_loss = last_loss + alpha * cl_loss
#     optimizer.zero_grad()
#     total_epoch_loss.backward()
#     optimizer.step()

#     # 记录日志
#     reward_curve.append(avg_reward)
#     loss_curve.append(last_loss.item())
#     safety_curve.append(safety_rate)

#     log_msg = f"[{datetime.now()}] Epoch {epoch} | PPO Loss: {last_loss.item():.4f} | Contrastive Loss: {cl_loss.item():.4f} | Total Loss: {total_epoch_loss.item():.4f} | Avg Reward: {avg_reward:.4f} | Safety Rate: {safety_rate:.2%}"
#     print(log_msg)
#     log_file.write(log_msg + "\n")

#     torch.save(policy_model.state_dict(), f"tiny_model_ppo_epoch{epoch}.pt")


log_file.close()

# ==== 绘制训练曲线 ====
os.makedirs("rlhf_pic", exist_ok=True)
# 图 1：Avg Reward
plt.figure(figsize=(10, 6))
plt.plot(reward_curve, label="Avg Reward")
plt.xlabel("Epoch")
plt.ylabel("Avg Reward")
plt.title("PPO + LoRA: Avg Reward Curve")
plt.grid(True)
plt.legend()
plt.savefig("rlhf_pic/ppo_lora_constra_reward.png")
plt.close()

# 图 2：Loss
plt.figure(figsize=(10, 6))
plt.plot(loss_curve, label="Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PPO + LoRA: Loss Curve")
plt.grid(True)
plt.legend()
plt.savefig("rlhf_pic/ppo_lora_loss.png")
plt.close()

# 图 3：Safety Rate
plt.figure(figsize=(10, 6))
plt.plot(safety_curve, label="Safety Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Safety Rate (%)")
plt.title("PPO + LoRA: Safety Rate Curve")
plt.grid(True)
plt.legend()
plt.savefig("rlhf_pic/ppo_loraconstra_safety.png")
plt.close()
#
plt.figure(figsize=(10, 6))
plt.plot(reward_curve, label="Avg Reward")
plt.plot(loss_curve, label="Loss")
plt.plot(safety_curve, label="Safety Rate")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("PPO Training Curve")
plt.legend()
plt.grid(True)
plt.savefig("rlhf_pic/ppo_lora_constra_training_curve.png")
plt.close()
