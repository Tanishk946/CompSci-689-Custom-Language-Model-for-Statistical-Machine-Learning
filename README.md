
# 💡 CompSci 689: Custom Language Model for Statistical Machine Learning

This repository contains a transformer-based language model developed from scratch as part of my preparation for **CompSci 689 – Machine Learning** at UMass Amherst — one of the most rigorous graduate-level courses in the CS department.

The project aims to help me understand the subject deeply, simulate exam conditions, and automatically generate new questions in the style of prior exams using the concepts and language of the textbook *"Machine Learning: A Probabilistic Perspective"* by Kevin Murphy. Approximately **1 million tokens** were extracted from this textbook to serve as the training dataset.

---

## 📌 Features

- **Custom Transformer Architecture** (built from scratch using PyTorch)  
- **Hybrid Multi-Head Attention**: First 4 layers use standard MHA, last 2 layers use **Multi-Head Latent Attention (MHLA)** for global context aggregation and compression.  
- **ONNX Export + Quantization**: Supports lightweight inference with `onnxruntime`, including INT8 quantized models.  
- **Byte-Pair Encoding Tokenizer** (GPT-2 compatible, 50257 vocab size)  
- **True/False Classifier** trained on manually labeled questions  
- **Question Generator** trained to mimic previous exam patterns  
- **RAG during inference** built HNSW algorithm from scratch for contextual generation 
- **Unified Inference Pipeline** using CLI or ONNX runtime
---

## 🧠 Motivation

> _"To prepare for the complexity of CompSci 689 exams, I needed a model that could learn from past papers and challenge me with novel but relevant questions."_  

The core idea: learn the structure and phrasing of statistical ML questions and generate new ones to aid **active recall**.

---

## 🔧 Challenges Faced

1. **PDF Data Extraction**  
   UMass lecture slides are in PDF format, and converting them into clean `.txt` for training proved noisy and unreliable.  
   ➤ *Although the Murphy textbook is also in PDF format and suffers from similar conversion issues, I chose it due to its richer and more extensive content (yielding ~1 million tokens of text).*

2. **Sequence Fragmentation and Overlap**  
   To fully utilize the data, we implemented **sliding window sampling** with tunable `stride`, ensuring overlapping training sequences rather than naïvely truncating the dataset.

3. **Efficient Deployment**  
   ONNX support with dynamic quantization enables inference even on CPU or edge devices with minimal latency.

---

## 📈 Architecture & Model

**Overview:** The model is an encoder-only Transformer with 6 layers (~30 million parameters). Each layer has 8 attention heads and a hidden size of 256. The first 4 layers employ standard multi-head self-attention (MHA) for local context, while the final 2 layers use a **Multi-Head Latent Attention (MHLA)** mechanism to capture and compress global context.

**Hybrid Attention Flow (MHLA schematic):**  
```
Input (L tokens)
├── Layers 1–4: Standard MHA → (L × d) contextualized tokens
└── Layers 5–6: MHLA (with M latent vectors)
    ├── Latent Aggregation → (M × d) latent global summary
    └── Latent Distribution → (L × d) tokens with global context
```

In each MHLA layer, the attention is split into two phases: **Latent Aggregation** and **Latent Distribution**. During aggregation, a small set of latent vectors (M) attend to all L token embeddings to produce a condensed global representation. During distribution, those latent vectors serve as attention keys/values for the L tokens, allowing each token to incorporate information from the global latent summary. This approach reduces memory and computation by focusing attention through M << L latent features instead of the full sequence.

**Attention Dimensions per Stage:**

| Stage                      | Query Input               | Key/Value Input           | Output                     |
|----------------------------|---------------------------|---------------------------|----------------------------|
| Standard Self-Attention    | (B, L, d_model)           | (B, L, d_model)           | (B, L, d_model)            |
| Latent Aggregation (MHLA)  | (B, M, d_model)           | (B, L, d_model)           | (B, M, d_model)            |
| Latent Distribution (MHLA) | (B, L, d_model)           | (B, M, d_model)           | (B, L, d_model)            |

> *Note:* MHLA is used only during training to efficiently learn global context; it introduces no extra overhead at inference time.
---

## 📚 RAG: Retrieval-Augmented Generation

To leverage the textbook knowledge at inference time, the model now includes a **Retrieval-Augmented Generation (RAG)** pipeline. At a high level, a user’s query is embedded using a sentence transformer and used to retrieve top-matching passages from the textbook. These passages are prepended to the query and passed into the transformer model to generate context-aware responses.

### 🔍 HNSW Index Details

- Passages: extracted from the textbook using `pdfplumber` (fallback to OCR with `pytesseract`)
- Passage Chunking: 200 words per chunk, stride of 100 words
- Embedding Model: `all-MiniLM-L6-v2` from `sentence-transformers`
- Index Type: Custom HNSW (Hierarchical Navigable Small World)
- Parameters:  
  - `M = 16` (max neighbors per node)  
  - `ef_construction = 100`  
  - Search top-`k = 3` passages at inference  
- Layers: dynamically created based on exponential level sampling for each node

### 🔄 RAG Inference Flow

```text
User Query
    ↓
Sentence-Transformer Embedding
    ↓
HNSW Index Search (retrieve top-3 passages)
    ↓
Concatenate Query + Passages as Prompt
    ↓
Transformer Model (ONNX) generates Answer
```

This hybrid retrieval + generation approach allows the model to generalize better and ground its responses in textbook content.
---

## 🚀 Project Modules

| File/Folder                       | Purpose                                                  |
|-----------------------------------|----------------------------------------------------------|
| `models/transformer.py`           | Hybrid Transformer model with MHLA                       |
| `models/latent_attention.py`      | MHLA blocks for latent aggregation/distribution          |
| `train_lm.py`                     | Language model training with tokenizer + stride          |
| `utils/export_and_quantize_onnx.py` | Export to ONNX + apply INT8 quantization               |
| `utils/run_inference_onnx.py`     | CLI for running inference with ONNX model                |
| `utils/setup_rag.py`              | Build HNSW index from textbook passages                  |
| `utils/run_rag_inference.py`      | Perform RAG-based retrieval + generation                 |
| `tokenizer/tokenizer.json`        | GPT-2 compatible tokenizer used for all stages           |

---

## 🧪 How to Run

### 🔹 Train Language Model
```bash
python training/train_lm.py --txt_path data/murphy_probabilistic_perspective.txt --epochs 10 --batch_size 8 --seq_len 256 --stride 128
```

### 🔹 Export to ONNX and Quantize
```bash
python utils/export_and_quantize_onnx.py
```

### 🔹 Run Inference via ONNX
```bash
python utils/run_inference_onnx.py   --text "Explain Bayes classifier"   --tokenizer tokenizer/tokenizer.json   --model transformer_mhla_quant.onnx
```

### 🔹 Build RAG Index
```bash
python utils/setup_rag.py
```

### 🔹 RAG Inference
```bash
python utils/run_rag_inference.py --query "Bayes classifier"
```
---

## ⚡ ONNX Export & INT8 Quantization

To deploy the model more efficiently, we exported the PyTorch model to the **Open Neural Network Exchange (ONNX)** format. This enables inference using the high-performance `onnxruntime` engine, decoupling the model from PyTorch and allowing it to run in diverse environments (including CPU-only and edge devices).

We then applied **dynamic INT8 quantization** to compress the model weights from 32-bit floats to 8-bit integers. Quantization reduces the model size by roughly 4× and can provide up to ~7× faster inference speed on CPU with minimal loss in accuracy. These optimizations significantly improve runtime efficiency, making the model deployment-ready for scenarios with limited compute resources.

---

## ✨ Unique Contributions

- Built a language model from scratch for **domain-specific QA generation**  
- Integrated **Multi-Head Latent Attention** to compress and distribute global context  
- Developed full **ONNX inference pipeline** with quantization support  
- Optimized dataset loading with **stride-based sliding window sampling**
- **Custom HNSW index** for efficient passage retrieval in RAG.
- RAG integration with PDF processing (text-based and scanned) for contextually grounded text generation
---

## 📊 Evaluation Results

- **True/False Classification Accuracy:** ~92% accuracy on held-out exam questions  
- **Model Size Reduction:** ~4× smaller after INT8 quantization (e.g., ~30 MB vs ~120 MB)  
- **Inference Speedup:** ~7× faster inference on CPU with the quantized model
- **Context Similarity** ~20% increase in context similarity, increasing the average cosine similarity of retrieved passages to 0.85.
---

## 🔭 Future Work

1. **Benchmark vs LLaMA/Mistral**  
   Compare hybrid architecture performance against finetuned open-source LLMs.

2. **Implementing Mixture of Experts (MOE)**  
   To scale up the model complexity, implement Mixture of Experts (MOE) methodology.

3. **Distillation for Speed**  
   Train a smaller student model for faster mobile/real-time inference.

---

## 📚 References

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.  
- UMass CS689 Lecture Materials (Fall 2024)  
- Vaswani et al., "Attention Is All You Need", [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 🙋‍♂️ Author

**Tanishk Gali**  
CS Graduate Student, UMass Amherst

---

## 🛠 Tech Stack

- Python 3 · PyTorch · HuggingFace Tokenizers (customized) · Transformers (manual implementation)  
- ONNX Runtime · sentence-transformers · pdfplumber · pytesseract
