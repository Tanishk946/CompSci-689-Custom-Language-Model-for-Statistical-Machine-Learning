import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader, Dataset
from models.transformer import TransformerModel
from utils.text_loader import load_text_file
from utils.tokenizer import tokenizer
import matplotlib.pyplot as plt

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_length):
        self.seq_length = seq_length
        self.samples = [
            torch.tensor(token_ids[i:i + seq_length], dtype=torch.long)
            for i in range(0, len(token_ids) - seq_length, seq_length)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = torch.cat([x[1:], torch.tensor([0])])
        return x, y

def train_language_model(txt_path, epochs=50, batch_size=16, seq_len=256, lr=1e-4):
    text = load_text_file(txt_path)
    token_ids = tokenizer.encode(text)
    vocab_size = tokenizer.vocab_size

    token_ids = token_ids[:(len(token_ids) // seq_len) * seq_len]

    dataset = TextDataset(token_ids, seq_length=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel(vocab_size)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/transformer_lm.pth")

    plt.figure()
    plt.plot(range(1, epochs+1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LM Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("models/loss_plot.png")
    plt.show()

if __name__ == "__main__":
    train_language_model(
        txt_path="data/murphy_probabilistic_perspective.txt",
        epochs=500,
        batch_size=8,
        seq_len=256,
        lr=1e-4
    )
