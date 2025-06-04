# file: generate.py
def generate_text(prompt, model, tokenizer, max_new_tokens=100, device="cuda"):
    import torch
    from torch.nn import functional as F

    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    model.eval()
    for _ in range(max_new_tokens):
        max_seq_len = 1024
        if input_ids.size(1) > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]

        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)



import random
import csv


def generate_samples(model, tokenizer, alpaca_data, output_path="model_eval_samples.csv"):
    sample_50 = random.sample(alpaca_data, 50)
    csv_rows = []

    for i, sample in enumerate(sample_50):
        prompt = f"Instruction: {sample['instruction']}\nInput: {sample['input']}\nResponse:"
        response = generate_text(prompt, model, tokenizer)
        csv_rows.append({
            "index": i + 1,
            "instruction": sample["instruction"],
            "input": sample["input"],
            "reference_output": sample["output"],
            "model_response": response,
            "accurate": ""
        })

    with open(output_path, mode="w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "instruction", "input", "reference_output", "model_response", "accurate"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved 50 evaluation samples to '{output_path}'. You can now open and annotate it (accurate: y/n).")


import csv

def evaluate_accuracy(csv_file="model_eval_samples.csv"):
    accurate_count = 0
    total = 0

    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["accurate"].strip().lower() == "y":
                accurate_count += 1
            total += 1

    accuracy = accurate_count / total * 100
    print(f"Accurate responses: {accurate_count}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")

    if accuracy > 60:
        print("Target met (accuracy > 60%)")
    else:
        print("Target not met (accuracy ≤ 60%)")




def test_example(model, tokenizer):
    instruction = "explain"
    input_text = "explain gravaity"
    prompt = f"Instruction:\n{instruction}\nInput:\n{input_text}\nOutput:\n"

    response = generate_text(prompt, model, tokenizer)
    print("=== 模型生成内容 ===")
    print(response)
