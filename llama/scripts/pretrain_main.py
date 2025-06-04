
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from tokenizers import Tokenizer as RawTokenizer

from llama.models.transformer import TinyTransformer
from llama.datasets.tokenized_dataset import TokenizedDataset
from llama.utils.seed import set_seed
from llama.utils.training import get_loss, evaluate
from llama.configs.pretrain import PRETRAIN_CONFIG
import pickle
from tqdm import tqdm  
def is_dist():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return local_rank

def cleanup_ddp():
    torch.distributed.destroy_process_group()

def main():
    use_ddp = is_dist()
    local_rank = setup_ddp() if use_ddp else 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(42 + local_rank)

    with open(PRETRAIN_CONFIG['tokenizer_path'], "rb") as f:
        raw_tokenizer = pickle.load(f)

    print("Tokenizer loaded!")
    vocab_size = raw_tokenizer.get_vocab_size()

    # 读取文本（首次构造仍需要）
    with open(PRETRAIN_CONFIG['train_data_path'], "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(PRETRAIN_CONFIG['valid_data_path'], "r", encoding="utf-8") as f:
        valid_text = f.read()

    # 使用缓存构造 TokenizedDataset
    train_dataset = TokenizedDataset(
        text=train_text,
        tokenizer=raw_tokenizer,
        block_size=PRETRAIN_CONFIG['block_size'],
        cache_path="./data/train_chunks.pt"
    )
    valid_dataset = TokenizedDataset(
        text=valid_text,
        tokenizer=raw_tokenizer,
        block_size=PRETRAIN_CONFIG['block_size'],
        cache_path="./data/valid_chunks.pt"
    )

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
    else:
        train_sampler = None
        valid_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=PRETRAIN_CONFIG['batch_size'], sampler=train_sampler, shuffle=not use_ddp, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=PRETRAIN_CONFIG['batch_size'], sampler=valid_sampler, num_workers=4, pin_memory=True)

    model = TinyTransformer(
        vocab_size,
        dim=PRETRAIN_CONFIG['model_dim'],
        n_heads=PRETRAIN_CONFIG['n_heads'],
        n_layers=PRETRAIN_CONFIG['n_layers'],
        block_size=PRETRAIN_CONFIG['block_size']
    )
    model = model.to(device)

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])


    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_CONFIG['lr'])
    scaler = GradScaler()

    save_dir = PRETRAIN_CONFIG['save_dir']
    pic_dir = PRETRAIN_CONFIG['pic_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pic_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, PRETRAIN_CONFIG['best_model_name'])
    best_val_ppl = float('inf')
    train_losses = []
    val_ppls = []

    for epoch in range(PRETRAIN_CONFIG['epochs']):
        model.train()
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)

        total_train_loss = 0
        count = 0

        # 添加进度条，仅主进程打印
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=use_ddp and local_rank != 0)

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = get_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            count += 1

            # 可选：动态显示 loss
            train_iter.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / count
        train_losses.append(avg_train_loss)

        val_ppl = evaluate(model, valid_loader, device)
        val_ppls.append(val_ppl)

        if not use_ddp or local_rank == 0:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val PPL: {val_ppl:.2f}")
            save_state = {
                'model_state_dict': model.module.state_dict() if use_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch + 1,
                'val_ppl': val_ppl,
            }
            # torch.save(save_state, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(save_state, best_model_path)
            if val_ppl < PRETRAIN_CONFIG['early_stop_ppl']:
                print(f"\U0001F389 提前停止：验证集 PPL 达到 {val_ppl:.2f} < {PRETRAIN_CONFIG['early_stop_ppl']}")
                break

    if not use_ddp or local_rank == 0:
        epochs_ran = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_ran, train_losses, label='Train Loss')
        plt.plot(epochs_ran, val_ppls, label='Val PPL')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss & Validation Perplexity')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(pic_dir, 'training_plot.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs_ran, val_ppls, label='Val PPL (zoomed)')
        plt.axhline(y=200, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('PPL')
        plt.title('Validation PPL (Zoomed View, <200)')
        plt.ylim(0, 200)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(pic_dir, 'ppl_zoomed.png'))
        plt.close()

    if use_ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main()