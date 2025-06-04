import json
from torch.utils.data import Dataset

class PKUSafeDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["prompt"]
