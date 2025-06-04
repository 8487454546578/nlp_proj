# import os
# import json
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from tqdm import tqdm
# from tokenizers import Tokenizer
# from torch.cuda.amp import autocast, GradScaler

# from llama.models.transformer import TinyTransformer
# from llama.scripts.generate import generate_text, generate_samples, evaluate_accuracy, test_example


# def is_ddp():
#     return "RANK" in os.environ and "WORLD_SIZE" in os.environ


# def setup_ddp():
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend="nccl")
#     return local_rank


# def cleanup_ddp():
#     if dist.is_initialized():
#         dist.destroy_process_group()


# class AlpacaDataset(Dataset):
#     def __init__(self, data, tokenizer, block_size=1024):
#         self.samples = []
#         for example in data:
#             prompt = f"Instruction:\n{example['instruction']}\nInput:\n{example['input']}\nOutput:\n{example['output']}"
#             ids = tokenizer.encode(prompt).ids
#             if len(ids) < block_size:
#                 ids += [0] * (block_size - len(ids))
#             else:
#                 ids = ids[:block_size]
#             self.samples.append(torch.tensor(ids))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         x = self.samples[idx][:-1]
#         y = self.samples[idx][1:]
#         return x, y


# def main():
#     ddp_mode = is_ddp()
#     local_rank = setup_ddp() if ddp_mode else 0
#     device = torch.device("cuda", local_rank)

#     tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
#     pad_token_id = tokenizer.token_to_id("<pad>")

#     checkpoint = torch.load("tiny_transformer_best.pth", map_location="cpu")
#     state_dict = checkpoint["model_state_dict"]
#     if any(k.startswith("module.") for k in state_dict):
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

#     vocab_size = state_dict["token_embedding.weight"].shape[0]
#     model = TinyTransformer(vocab_size=vocab_size).to(device)
#     model.load_state_dict(state_dict)

#     # === 冻结所有参数，仅微调最后两层 block + norm + lm_head ===
#     for param in model.parameters():
#         param.requires_grad = False
#     for i in [-2, -1]:
#         for param in model.blocks[i].parameters():
#             param.requires_grad = True
#     for name, param in model.named_parameters():
#         if "norm" in name or "lm_head" in name:
#             param.requires_grad = True

#     if ddp_mode:
#         model = DDP(model, device_ids=[local_rank])

#     if local_rank == 0:
#         print("Trainable parameters:")
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 print(f"  ✓ {name}")

#     with open("./data/alpaca-1k.json", "r", encoding="utf-8") as f:
#         alpaca_data = [json.loads(line) for line in f]

#     dataset = AlpacaDataset(alpaca_data, tokenizer)
#     sampler = DistributedSampler(dataset) if ddp_mode else None
#     loader = DataLoader(dataset, batch_size=4, sampler=sampler, shuffle=not ddp_mode)

#     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
#     loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
#     scaler = GradScaler()

#     os.makedirs("checkpoints", exist_ok=True)
#     os.makedirs("finetune", exist_ok=True)

#     model.train()
#     for epoch in range(10):
#         if ddp_mode:
#             sampler.set_epoch(epoch)
#         total_loss = 0.0

#         for x, y in tqdm(loader, disable=local_rank != 0):
#             x, y = x.to(device), y.to(device)

#             optimizer.zero_grad()
#             with autocast():
#                 logits = model(x)
#                 B, T, V = logits.shape
#                 loss = loss_fn(logits.view(B * T, V), y.view(B * T))

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item()

#         if local_rank == 0:
#             avg_loss = total_loss / len(loader)
#             print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

#             # === 保存 checkpoint ===
#             checkpoint = {
#                 "epoch": epoch + 1,
#                 "model_state_dict": (model.module if ddp_mode else model).state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "scaler_state_dict": scaler.state_dict(),
#                 "loss": avg_loss,
#             }
#             ckpt_path = f"checkpoints/finetune/epoch_{epoch+1}.pt"
#             torch.save(checkpoint, ckpt_path)
#             print(f"✅ Checkpoint saved to {ckpt_path}")

#     # === 保存最终模型 state_dict（用于推理）===
#     if local_rank == 0:
#         torch.save((model.module if ddp_mode else model).state_dict(),
#                    "finetune/tiny_model_finetuned_alpaca.pt")
#         print("✅ Final model saved to finetune/tiny_model_finetuned_alpaca.pt")

#     if ddp_mode:
#         cleanup_ddp()



# if __name__ == "__main__":
#     main()
import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from tokenizers import Tokenizer
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

from llama.models.transformer import TinyTransformer

def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, block_size=1024):
        self.samples = []
        for example in data:
            prompt = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n"
            )
            response = example['output']
            ids = tokenizer.encode(prompt + response).ids
            if len(ids) < block_size:
                ids += [0] * (block_size - len(ids))
            else:
                ids = ids[:block_size]
            self.samples.append(torch.tensor(ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = ids[:-1]
        y = ids[1:]
        return x, y

@torch.no_grad()
def evaluate_loss(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), y.view(B * T))
        total_loss += loss.item()
        total_batches += 1
    return total_loss / total_batches

def main():
    ddp_mode = is_ddp()
    local_rank = setup_ddp() if ddp_mode else 0
    device = torch.device("cuda", local_rank)

    tokenizer = Tokenizer.from_file("./data/tiny_tokenizer.json")
    pad_token_id = tokenizer.token_to_id("<pad>")

    checkpoint = torch.load("tiny_transformer_best.pth", map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    model = TinyTransformer(vocab_size=vocab_size).to(device)
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False
    for i in [-2, -1]:
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    for name, param in model.named_parameters():
        if "norm" in name or "lm_head" in name:
            param.requires_grad = True

    if ddp_mode:
        model = DDP(model, device_ids=[local_rank])

    if local_rank == 0:
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  ✓ {name}")

    with open("./data/alpaca-1k.json", "r", encoding="utf-8") as f:
        alpaca_data = [json.loads(line) for line in f]

    train_data, val_data = train_test_split(alpaca_data, test_size=0.1, random_state=42)

    train_dataset = AlpacaDataset(train_data, tokenizer)
    val_dataset = AlpacaDataset(val_data, tokenizer)

    train_sampler = DistributedSampler(train_dataset) if ddp_mode else None
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, shuffle=not ddp_mode)
    val_loader = DataLoader(val_dataset, batch_size=4)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    scaler = GradScaler()

    os.makedirs("checkpoints/finetune", exist_ok=True)
    os.makedirs("finetune", exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 3
    model.train()

    for epoch in range(40):
        if ddp_mode:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        for x, y in tqdm(train_loader, disable=local_rank != 0):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                B, T, V = logits.shape
                loss = loss_fn(logits.view(B * T, V), y.view(B * T))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if local_rank == 0 else 0.0
        val_loss = evaluate_loss(model.module if ddp_mode else model, val_loader, loss_fn, device)

        if local_rank == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": (model.module if ddp_mode else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            }
            ckpt_path = f"checkpoints/finetune/epoch_{epoch+1}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"✅ Checkpoint saved to {ckpt_path}")

        should_stop = torch.tensor([0], device=device)
        if local_rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save((model.module if ddp_mode else model).state_dict(),
                           "finetune/best_model.pt")
                print("🌟 New best model saved.")
            else:
                epochs_no_improve += 1
                print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")
                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    should_stop[0] = 1

        if ddp_mode:
            dist.broadcast(should_stop, src=0)

        if should_stop.item() == 1:
            break

    if local_rank == 0:
        torch.save((model.module if ddp_mode else model).state_dict(),
                   "finetune/tiny_model_finetuned_alpaca.pt")
        print("✅ Final model saved to finetune/tiny_model_finetuned_alpaca.pt")

    if ddp_mode:
        dist.barrier()
        cleanup_ddp()

if __name__ == "__main__":
    main()
