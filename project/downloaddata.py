import os
from datasets import load_dataset, load_from_disk, DatasetDict
# 第一次联网下载并保存

print("HF_HOME =", os.getenv("HF_HOME"))

dataset = load_dataset("roneneldan/TinyStories")

# 保存到本地（包含所有 split：train/validation）
dataset.save_to_disk("data/tinystories")
dataset = load_dataset("data/TinyStories")