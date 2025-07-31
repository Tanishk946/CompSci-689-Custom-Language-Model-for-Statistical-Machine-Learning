import numpy as np
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
from tokenizers import Tokenizer
import argparse
from typing import List, Tuple

class HNSWIndex:
    def __init__(self, M: int = 16, ef_construction: int = 100):
        self.M = M
        self.ef_construction = ef_construction
        self.layers = []
        self.entry_point = None
        self.max_level = 0

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def insert(self, embedding: np.ndarray, text: str, node_id: str):
        level = int(-np.log(np.random.random()) * self.M)
        self.max_level = max(self.max_level, level)
        while len(self.layers) <= level:
            self.layers.append([])
        node = (embedding, text, node_id)
        current_entry = self.entry_point
        current_level = self.max_level
        for l in range(self.max_level, level - 1, -1):
            if current_entry is None:
                break
            current_entry = self._find_closest(current_entry, embedding, l)
        for l in range(min(level, self.max_level) + 1):
            neighbors = []
            if current_entry:
                neighbors = self._select_neighbors(embedding, l)
            self.layers[l].append((node, neighbors))
            if l == 0 and self.entry_point is None:
                self.entry_point = node_id

    def _find_closest(self, entry_node_id: str, query: np.ndarray, layer: int) -> str:
        best_node_id = entry_node_id
        best_sim = -1.0
        for node, _ in self.layers[layer]:
            sim = self.cosine_similarity(query, node[0])
            if sim > best_sim:
                best_sim = sim
                best_node_id = node[2]
        return best_node_id

    def _select_neighbors(self, query: np.ndarray, layer: int) -> List[str]:
        candidates = []
        for node, _ in self.layers[layer]:
            sim = self.cosine_similarity(query, node[0])
            candidates.append((sim, node[2]))
        candidates.sort(reverse=True)
        return [nid for _, nid in candidates[:self.M]]

    def search(self, query: np.ndarray, k: int = 3, ef_search: int = 64) -> List[Tuple[str, float]]:
        if not self.layers[0]:
            return []
        current_entry = self.entry_point
        for l in range(self.max_level, -1, -1):
            current_entry = self._find_closest(current_entry, query, l)
        candidates = []
        for node, _ in self.layers[0]:
            sim = self.cosine_similarity(query, node[0])
            candidates.append((sim, node[1], node[2]))
        candidates.sort(reverse=True)
        return [(text, sim) for sim, text, _ in candidates[:k]]

    @classmethod
    def load(cls, index_path: str):
        data = np.load(index_path, allow_pickle=True).item()
        hnsw = cls(M=16, ef_construction=100)
        hnsw.layers = data['layers']
        hnsw.entry_point = data['entry_point']
        hnsw.max_level = data['max_level']
        return hnsw

def generate_text(ort_session, tokenizer, input_text: str, max_length: int = 50, max_seq_len: int = 256):
    """Generate text using ONNX model with greedy decoding, maintaining max_seq_len."""
    input_ids = tokenizer.encode(input_text).ids
    # Truncate input to max_seq_len to match model expectations
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        print(f"Warning: Input truncated to {max_seq_len} tokens.")
    # Pad input if shorter than max_seq_len
    if len(input_ids) < max_seq_len:
        pad_token_id = tokenizer.token_to_id("[PAD]") or 0
        input_ids = input_ids + [pad_token_id] * (max_seq_len - len(input_ids))
    generated_ids = input_ids.copy()
    for _ in range(max_length):
        inputs = np.array([generated_ids], dtype=np.int64)
        # Verify input shape compatibility
        expected_shape = ort_session.get_inputs()[0].shape
        if len(inputs.shape) != len(expected_shape) or inputs.shape[0] != 1 or inputs.shape[1] != max_seq_len:
            print(f"Input shape {inputs.shape} does not match expected {expected_shape}")
            raise ValueError(f"Input shape mismatch: got {inputs.shape}, expected [1, {max_seq_len}]")
        outputs = ort_session.run(None, {"input": inputs})
        logits = outputs[0][0, -1, :]
        next_token_id = np.argmax(logits)
        generated_ids.append(next_token_id)
        # Truncate to max_seq_len after appending to maintain shape
        if len(generated_ids) > max_seq_len:
            generated_ids = generated_ids[:max_seq_len]
        if next_token_id == tokenizer.token_to_id("[EOS]"):
            break
    return tokenizer.decode(generated_ids)

def main(args):
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ort_session = ort.InferenceSession(args.model)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    hnsw = HNSWIndex.load(args.index)
    query_embedding = embedder.encode([args.query])[0]
    passages = hnsw.search(query_embedding, k=args.k)
    # Limit each passage to 100 words to reduce prompt length
    context = " ".join([p[0][:100] for p in passages])
    prompt = f"Question about {args.query}: {context}"
    generated_question = generate_text(ort_session, tokenizer, prompt, max_length=50, max_seq_len=256)
    print(f"Generated Text: {generated_question}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG inference for question generation.")
    parser.add_argument("--query", type=str, required=True, help="Query for generating the question (e.g., 'Bayes classifier')")
    parser.add_argument("--model", type=str, default="transformer_mhla_quant.onnx", help="Path to the ONNX model")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to the tokenizer")
    parser.add_argument("--index", type=str, default="hnsw_index.npy", help="Path to the HNSW index")
    parser.add_argument("--k", type=int, default=3, help="Number of passages to retrieve")
    args = parser.parse_args()
    main(args)