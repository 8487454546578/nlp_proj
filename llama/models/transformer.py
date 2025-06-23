
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / math.sqrt(x.shape[-1])
        return self.weight * x / (rms + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden)
        self.w2 = nn.Linear(dim_in, dim_hidden)
        self.w3 = nn.Linear(dim_hidden, dim_in)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

def apply_rope(x):
    """
    x: [B, T, H, D] - 应为 q 或 k
    返回: 应用 RoPE 后的向量，维度保持不变
    """
    B, T, H, D = x.shape
    device = x.device
    assert D % 2 == 0, "head_dim 必须是偶数"

    theta = 10000 ** (-torch.arange(0, D, 2, device=device).float() / D)  # [D//2]
    pos = torch.arange(T, device=device).float()                         # [T]
    freqs = torch.einsum("i,j->ij", pos, theta)                          # [T, D//2]

    sin, cos = freqs.sin(), freqs.cos()
    sin = sin[None, :, None, :]  # [1, T, 1, D//2]
    cos = cos[None, :, None, :]

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rope = torch.zeros_like(x)
    x_rope[..., ::2] = x1 * cos - x2 * sin
    x_rope[..., 1::2] = x1 * sin + x2 * cos
    return x_rope


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, self.n_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # (B, H, T, D)

        q, k = apply_rope(q), apply_rope(k)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)  # (B, H, T, D)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 2)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


#decoder-only ,简化版LLaMA
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, dim=1024, n_heads=32, n_layers=16, block_size=1024):
        #目前最好是1024 32 16 1024
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, x, attention_mask=None):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.norm(x)
        return self.lm_head(x)