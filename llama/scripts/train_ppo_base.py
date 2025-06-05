import os
import sys
import json
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from datetime import datetime
import matplotlib.pyplot as plt

# 加载本地模块
sys.path.append(os.path.abspath(".."))
from llama.models.transformer import TinyTransformer
from llama.models.reward_model import RewardModel
from llama.datasets.pku_safe import PKUSafeDataset
from llama.utils.evaluate_safety import evaluate_safety_rate
from llama.utils.generation import generate_response
from llama.utils.loss import ppo_loss
from llama.utils.reward_utils import compute_reward

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 tokenizer
tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

# 加载策略模型
policy_model = TinyTransformer(vocab_size=vocab_size).to(device)
policy_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
policy_model.train()

# 加载奖励模型
base_model = TinyTransformer(vocab_size=vocab_size).to(device)
reward_model = RewardModel(base_model).to(device)
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()

# 准备数据集
train_dataset = PKUSafeDataset("./data/train.jsonl")
test_dataset = PKUSafeDataset("./data/test.jsonl")

# 限制训练/测试数据规模（加速调试）
subset_dataset = torch.utils.data.Subset(train_dataset, list(range(min(1000, len(train_dataset)))))
test_subset = [test_dataset[i] for i in range(min(len(test_dataset), 100))]

train_loader = DataLoader(subset_dataset, batch_size=8, shuffle=True)
print("训练集样本数量:", len(subset_dataset))
print("测试子集样本数量:", len(test_subset))

# PPO 优化器
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

# 日志文件与指标记录
log_file = open("ppo_training_log.txt", "a")
reward_curve = []
loss_curve = []
safety_curve = []

# PPO 训练主循环
for epoch in range(20):
    total_loss, total_reward = 0.0, 0.0
    for prompts in tqdm(train_loader, desc=f"Epoch {epoch}"):
        prompt = prompts[0]  # 每 batch 仅取一个 prompt（可扩展为 batch 生成）

        # 生成响应
        response = generate_response(policy_model, prompt, tokenizer)

        # 计算 reward
        reward = compute_reward(prompt, response, reward_model=reward_model, tokenizer=tokenizer)
        total_reward += reward

        # 编码输入
        text = prompt + " " + response
        encoding = tokenizer.encode(text)
        input_ids = torch.tensor([encoding.ids], device=device)

        # 计算旧策略 log prob
        with torch.no_grad():
            logits = policy_model(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            old_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

        # PPO 更新
        advantage = torch.tensor([reward], device=device)
        loss = ppo_loss(policy_model, old_log_probs, input_ids, advantage)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估安全率
    safety_rate = evaluate_safety_rate(policy_model, reward_model, tokenizer, test_dataset=test_subset)

    # 记录当前 epoch 指标
    avg_reward = total_reward / len(train_loader)
    avg_loss = total_loss / len(train_loader)
    reward_curve.append(avg_reward)
    loss_curve.append(avg_loss)
    safety_curve.append(safety_rate)

    # 输出日志
    log_msg = f"[{datetime.now()}] Epoch {epoch} | Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.4f} | Safety Rate: {safety_rate:.2%}"
    print(log_msg)
    log_file.write(log_msg + "\n")

    # 条件保存模型
    if epoch % 5 == 0 or safety_rate > 0.8:
        torch.save(policy_model.state_dict(), f"tiny_model_ppo_epoch{epoch}.pt")

log_file.close()

# 绘制训练曲线
os.makedirs("rlhf_pic", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(reward_curve, label="Avg Reward")
plt.plot(loss_curve, label="Loss")
plt.plot(safety_curve, label="Safety Rate")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("PPO Training Curve")
plt.legend()
plt.grid(True)
plt.savefig("rlhf_pic/ppo_training_curve.png")
plt.close()
