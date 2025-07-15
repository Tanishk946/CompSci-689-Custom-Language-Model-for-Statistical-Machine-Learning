import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader
from models.transformer import TransformerModel
from utils.tokenizer import tokenizer
from utils.text_loader import load_text_file

class QuestionDataset(Dataset):
    def __init__(self, path, seq_len=128):
        text = load_text_file(path)
        # split by lines (one question per line)
        questions = [line.strip() for line in text.split("\n") if len(line.strip()) > 10]
        joined = " ".join(questions)
        token_ids = tokenizer.encode(joined)
        token_ids = token_ids[:(len(token_ids) // seq_len) * seq_len]
        self.data = torch.tensor(token_ids).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y

def train_question_gen_model(txt_path, seq_len=128, batch_size=8, epochs=500, lr=1e-4):
    dataset = QuestionDataset(txt_path, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerModel(vocab_size=tokenizer.vocab_size, max_seq_len=seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/question_gen_model.pth")

if __name__ == "__main__":
    train_question_gen_model("data/questions.txt", seq_len=128)
