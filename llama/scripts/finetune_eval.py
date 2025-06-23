import os 
import torch
import json
import random
from tokenizers import Tokenizer

from llama.models.transformer import TinyTransformer
from llama.scripts.generate import generate_text, generate_samples, evaluate_accuracy, test_example

def load_model(model_path, tokenizer_path, device="cuda"):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = TinyTransformer(vocab_size=vocab_size)
    state_dict = torch.load(model_path, map_location=device)

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "finetune/best_model.pt"
    tokenizer_path = "./data/tiny_tokenizer.json"
    data_path = "./data/alpaca-1k.json"

    print("== 加载模型和分词器 ==")
    model, tokenizer = load_model(model_path, tokenizer_path, device)

    print("== 加载数据并抽样 ==")
    with open(data_path, "r", encoding="utf-8") as f:
        alpaca_data = [json.loads(line) for line in f]
    sampled_data = random.sample(alpaca_data, 50)

    # print("== 生成样本 ==")
    # generate_samples(model, tokenizer, sampled_data, output_path="finetune/model_eval_samples.csv")

    # print("== 精度评估 ==")
    # evaluate_accuracy(csv_file="finetune/model_eval_samples.csv")

    print("== 示例测试 ==")
    test_example(model, tokenizer)

if __name__ == "__main__":
    main()
