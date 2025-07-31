import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pdfplumber
import re
from pdf2image import convert_from_path
import cv2
from PIL import Image
import pytesseract
import random

class HNSWIndex:
    def __init__(self, M: int = 16, ef_construction: int = 100):
        self.M = M  # Max neighbors per node
        self.ef_construction = ef_construction  # Neighbors to explore during insertion
        self.layers = [[]]  # List of layers, each containing nodes [(embedding, text, id), [neighbor_ids]]
        self.entry_point = None  # Node to start search
        self.max_level = 0

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def insert(self, embedding: np.ndarray, text: str, node_id: str):
        """Insert a new embedding into the HNSW index."""
        # Assign random level based on exponential decay
        level = int(-np.log(random.random()) * self.M)
        self.max_level = max(self.max_level, level)
        while len(self.layers) <= level:
            self.layers.append([])

        node = (embedding, text, node_id)
        current_entry = self.entry_point
        current_level = self.max_level

        # Navigate to insertion layer
        for l in range(self.max_level, level - 1, -1):
            if current_entry is None:
                break
            current_entry = self._find_closest(current_entry, embedding, l)

        # Insert node in all layers from 0 to level
        for l in range(min(level, self.max_level) + 1):
            neighbors = []
            if current_entry:
                neighbors = self._select_neighbors(embedding, l)
            self.layers[l].append((node, neighbors))
            if l == 0 and self.entry_point is None:
                self.entry_point = node_id

    def _find_closest(self, entry_node_id: str, query: np.ndarray, layer: int) -> str:
        """Find the closest node to query in the given layer."""
        best_node_id = entry_node_id
        best_sim = -1.0
        for node, _ in self.layers[layer]:
            sim = self.cosine_similarity(query, node[0])
            if sim > best_sim:
                best_sim = sim
                best_node_id = node[2]
        return best_node_id

    def _select_neighbors(self, query: np.ndarray, layer: int) -> List[str]:
        """Select up to M nearest neighbors in the given layer."""
        candidates = []
        for node, _ in self.layers[layer]:
            sim = self.cosine_similarity(query, node[0])
            candidates.append((sim, node[2]))
        candidates.sort(reverse=True)
        return [nid for _, nid in candidates[:self.M]]

    def search(self, query: np.ndarray, k: int = 3, ef_search: int = 64) -> List[Tuple[str, float]]:
        """Search for top-k nearest neighbors."""
        if not self.layers[0]:
            return []
        current_entry = self.entry_point
        for l in range(self.max_level, -1, -1):
            current_entry = self._find_closest(current_entry, query, l)
        
        # At bottom layer, collect ef_search candidates
        candidates = []
        for node, _ in self.layers[0]:
            sim = self.cosine_similarity(query, node[0])
            candidates.append((sim, node[1], node[2]))
        candidates.sort(reverse=True)
        return [(text, sim) for sim, text, _ in candidates[:k]]

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber; fall back to pytesseract for scanned pages."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    page_text = re.sub(r'\n\s*\n+', '\n', page_text)
                    page_text = re.sub(r'Page \d+', '', page_text)
                    page_text = re.sub(r'^\s*Chapter \d+.*$', '', page_text, flags=re.MULTILINE)
                    page_text = page_text.strip()
                    if page_text:
                        text += page_text + " "
    except Exception as e:
        print(f"pdfplumber error: {e}. Falling back to OCR with pytesseract.")
    
    if not text:
        try:
            pages = convert_from_path(pdf_path, dpi=300)
            for page_number, page_image in enumerate(pages, start=1):
                image = np.array(page_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                processed_image = preprocess_image_for_ocr(image)
                pil_image = Image.fromarray(processed_image)
                page_text = pytesseract.image_to_string(pil_image, config='--psm 6')
                page_text = re.sub(r'\n\s*\n+', '\n', page_text)
                page_text = page_text.strip()
                if page_text:
                    text += page_text + " "
                print(f"Processed page {page_number} with OCR.")
        except Exception as e:
            print(f"OCR error: {e}")
    return text

def preprocess_textbook(pdf_path: str, chunk_size: int = 200, stride: int = 100) -> List[str]:
    """Split textbook PDF into passages for RAG knowledge base."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError(f"No text extracted from {pdf_path}")
    words = text.split()
    passages = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), stride)]
    return [p for p in passages if len(p.strip()) > 50]

def setup_hnsw_index(passages: List[str], index_path: str = "hnsw_index.npy"):
    """Create and save HNSW index for passages."""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(passages, batch_size=32, show_progress_bar=True)
    hnsw = HNSWIndex(M=16, ef_construction=100)
    for i, (emb, text) in enumerate(zip(embeddings, passages)):
        hnsw.insert(emb, text, f"passage_{i}")
    # Save index (simplified as numpy arrays for embeddings and metadata)
    np.save(index_path, {'layers': hnsw.layers, 'entry_point': hnsw.entry_point, 'max_level': hnsw.max_level})
    print(f"Indexed {len(passages)} passages in HNSW index, saved to {index_path}.")

if __name__ == "__main__":
    pdf_path = "data/murphy_probabilistic_perspective.pdf"
    passages = preprocess_textbook(pdf_path)
    setup_hnsw_index(passages)