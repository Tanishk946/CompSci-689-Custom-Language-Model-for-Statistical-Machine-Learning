import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import torch
from models.transformer import TransformerModel, TransformerClassifier
from utils.tokenizer import tokenizer

# Load language model
def load_lm_model(path="models/transformer_lm.pth", seq_len=512):
    model = TransformerModel(vocab_size=tokenizer.vocab_size, max_seq_len=seq_len)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# Load true/false classifier
def load_tf_model(path="models/tf_classifier.pth"):
    base = TransformerModel(vocab_size=tokenizer.vocab_size)
    model = TransformerClassifier(base_model=base, hidden_dim=256)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# Load question generation model
def load_qg_model(path="models/question_gen_model.pth", seq_len=128):
    model = TransformerModel(vocab_size=tokenizer.vocab_size, max_seq_len=seq_len)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def generate_text(model, prompt, max_new_tokens=50):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token = torch.argmax(logits[0, -1]).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
    return tokenizer.decode(input_ids[0].tolist())

def classify_true_false(model, statement):
    tokens = tokenizer.encode(statement)
    input_ids = torch.tensor(tokens[:128]).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids)
        pred = torch.argmax(logits, dim=-1).item()
    return "True" if pred == 1 else "False"

if __name__ == "__main__":
    print("Choose mode: [1] LM Completion  [2] True/False  [3] Question Gen")
    mode = input("Enter option number: ").strip()

    if mode == "1":
        model = load_lm_model()
        prompt = input("Prompt > ")
        output = generate_text(model, prompt)
        print("LM Output:", output)

    elif mode == "2":
        model = load_tf_model()
        stmt = input("Enter statement to classify as True or False: ")
        result = classify_true_false(model, stmt)
        print("Prediction:", result)

    elif mode == "3":
        model = load_qg_model()
        topic = input("Start of question > ")
        question = generate_text(model, topic)
        print("Generated Question:", question)

    else:
        print("Invalid option.")
