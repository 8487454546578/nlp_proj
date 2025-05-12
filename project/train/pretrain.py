import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.transformer import TransformerLM
from data.download_dataset import load_local_dataset

# 初始化
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=1024)

def collate_fn(batch):
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    labels = input_ids.clone()
    return input_ids, attention_mask, labels

def main():
    dataset = load_local_dataset("data/tinystories", ["train", "validation"])
    dataset = dataset.map(tokenize, batched=True).remove_columns(['text'])

    train_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset['validation'], batch_size=8, collate_fn=collate_fn)

    model = TransformerLM(vocab_size=tokenizer.vocab_size).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0
        for input_ids, _, labels in train_loader:
            input_ids, labels = input_ids.cuda(), labels.cuda()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}")

        # 验证困惑度
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, _, labels in val_loader:
                input_ids, labels = input_ids.cuda(), labels.cuda()
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
        ppl = torch.exp(torch.tensor(val_loss / len(val_loader)))
        print(f"Val PPL: {ppl:.2f}")

if __name__ == '__main__':
    main()
