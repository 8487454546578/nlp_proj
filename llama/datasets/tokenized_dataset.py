import os
import torch
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, text=None, tokenizer=None, block_size=1024, cache_path=None):
        self.block_size = block_size

        if cache_path is not None and os.path.exists(cache_path):
            # ✅ 从缓存加载
            self.chunks = torch.load(cache_path)
            print(f"[TokenizedDataset] 加载缓存：{cache_path} ({len(self.chunks)} 个样本)")
        else:
            assert text is not None and tokenizer is not None, "首次构造必须提供 text 和 tokenizer"
            self.chunks = []

            stories = text.split('<|endoftext|>')
            for story in stories:
                encoded = tokenizer.encode(story + '<|endoftext|>').ids
                if len(encoded) < block_size + 1:
                    continue
                for i in range(0, len(encoded) - block_size, block_size):
                    chunk = encoded[i:i + block_size + 1]
                    self.chunks.append(chunk)

            print(f"[TokenizedDataset] 生成新数据：{len(self.chunks)} 个样本")
            if cache_path is not None:
                torch.save(self.chunks, cache_path)
                print(f"[TokenizedDataset] 缓存已保存到：{cache_path}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
