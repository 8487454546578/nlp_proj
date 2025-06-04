import os
import torch
import json
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

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

# 加载模型
policy_model = TinyTransformer(vocab_size=vocab_size).to(device)
policy_model.load_state_dict(torch.load("./finetune/tiny_model_finetuned_alpaca.pt"))
policy_model.train()

base_model = TinyTransformer(vocab_size=vocab_size).to(device)
reward_model = RewardModel(base_model).to(device)
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()

# 准备数据
train_dataset = PKUSafeDataset("./data/train.jsonl")
subset_dataset = torch.utils.data.Subset(train_dataset, list(range(10)))
train_loader = DataLoader(subset_dataset, batch_size=1, shuffle=True)

# PPO 优化器
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

# 简化版 PPO 训练主循环
for epoch in range(1):  # 可调整轮数
    for prompt in tqdm(train_loader):
        prompt = prompt[0]  # 解包 prompt

        # 生成响应
        response = generate_response(policy_model, prompt)

        # 编码输入
        text = prompt + " " + response
        encoding = tokenizer.encode(text)
        input_ids = torch.tensor([encoding.ids], device=device)

        # 计算 reward 和 advantage
        reward = compute_reward(prompt, response)
        baseline = 0
        advantage = torch.tensor([reward - baseline], device=device)

        # 计算旧策略的 log prob
        with torch.no_grad():
            logits = policy_model(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            old_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

        # 计算 PPO 损失并更新策略
        loss = ppo_loss(policy_model, old_log_probs, input_ids, advantage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每轮保存模型并评估安全率
    torch.save(policy_model.state_dict(), f"tiny_model_ppo_epoch{epoch}.pt")
    evaluate_safety_rate(policy_model, reward_model, sample_size=100, threshold=0.5)
