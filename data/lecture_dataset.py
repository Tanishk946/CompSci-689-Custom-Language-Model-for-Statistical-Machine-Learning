import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_length):
        self.seq_length = seq_length
        self.data = token_ids
        self.num_tokens = len(self.data)

    def __len__(self):
        return max(0, self.num_tokens - self.seq_length)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1: idx + 1 + self.seq_length], dtype=torch.long)
        return x, y

class TrueFalseDataset(Dataset):
    def __init__(self, statements, labels):
        assert len(statements) == len(labels)
        self.statements = statements
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.statements[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
