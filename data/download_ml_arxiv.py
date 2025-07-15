from datasets import load_dataset
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_ml_texts(max_samples=10000):
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    texts = ds["abstract"][:max_samples]
    return texts

def tokenize_ml_texts(texts):
    token_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        token_ids.extend(ids)
    return token_ids

if __name__ == "__main__":
    print("Downloading ML abstracts...")
    texts = get_ml_texts()
    print("Tokenizing...")
    token_ids = tokenize_ml_texts(texts)
    print(f"Prepared {len(token_ids)} tokens.")
