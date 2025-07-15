import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader
from models.transformer import TransformerClassifier, TransformerModel
from utils.tokenizer import tokenizer

class TFDataset(Dataset):
    def __init__(self, texts, labels, max_len=128):
        self.inputs = [torch.tensor(tokenizer.encode(t)[:max_len]) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], torch.tensor(self.labels[idx])

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    labels = torch.stack(labels)
    return inputs, labels

def load_tf_dataset(path):
    statements, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if '\t' in line:
                s, l = line.strip().rsplit("\t", 1)
            else:
                s, l = line.strip().rsplit("\t", 1)
            statements.append(s)
            labels.append(int(l))
    return statements, labels

def train_true_false_classifier(statements, labels, epochs=5):
    dataset = TFDataset(statements, labels)
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    base_model = TransformerModel(vocab_size=tokenizer.vocab_size)
    model = TransformerClassifier(base_model, hidden_dim=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/tf_classifier.pth")

if __name__ == "__main__":
    statements, labels = load_tf_dataset("data/tf_statements_dataset.txt")
    train_true_false_classifier(statements, labels, epochs=250)
