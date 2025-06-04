import torch
import torch.nn.functional as F
import math

def get_loss(logits, targets):
    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)
    return F.cross_entropy(logits, targets)

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    count = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = get_loss(logits, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return math.exp(total_loss / count)
