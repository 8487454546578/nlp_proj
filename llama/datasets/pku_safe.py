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

class PKUSafePairDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "response_0": item["response_0"],
            "response_1": item["response_1"],
            "safer_response_id": item.get("safer_response_id", 1)
        }

    def __len__(self):
        return len(self.data)
