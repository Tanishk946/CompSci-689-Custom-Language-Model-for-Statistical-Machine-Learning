# 💡 CompSci 689: Custom Language Model for Statistical Machine Learning

This repository contains a transformer-based language model developed from scratch as part of my preparation for **CompSci 689 – Machine Learning** at UMass Amherst — one of the most rigorous graduate-level courses in the CS department.

The project aims to help me understand the subject deeply, simulate exam conditions, and automatically generate new questions in the style of prior exams using the concepts and language of the textbook *"Machine Learning: A Probabilistic Perspective"* by Kevin Murphy.

---

## 📌 Features

- **Custom Transformer Architecture** (built from scratch using PyTorch)
- **Fine-tuned Language Model** on the full Murphy textbook
- **True/False Classifier** trained on manually labeled questions
- **Question Generator** trained to mimic previous exam patterns
- **Unified Inference Script** to test all three modes: completion, classification, generation

---

## 🧠 Motivation

> _"To prepare for the complexity of CompSci 689 exams, I needed a model that could learn from past papers and challenge me with novel but relevant questions."_  

The core idea: learn the structure and phrasing of statistical ML questions and generate new ones to **simulate exam pressure** and aid **active recall**.

---

## 🔧 Challenges Faced

1. **PDF Data Extraction**  
   UMass lecture slides are in PDF format, and converting them into clean `.txt` for training proved noisy and unreliable.  
   ➤ *Although the Murphy textbook is also in PDF format and suffers from similar conversion issues, I chose it due to its richer and more extensive content compared to the lecture slides.*

2. **Textbook-Level Complexity**  
   Training on Kevin Murphy’s probabilistic ML textbook required higher compute resources than standard fine-tuning tasks.

---

## 📈 Architecture & Model

- **Transformer Encoder Only**
- 6 Layers · 8 Heads · 256-D Embeddings
- **~30M Parameters**
- Byte-Pair Encoding tokenizer (50257 vocab size, GPT-2 style)

---

## 🚀 Project Modules

| File                     | Purpose                               |
|--------------------------|----------------------------------------|
| `train_lm.py`            | Language model fine-tuning             |
| `train_truefalse.py`     | True/False classification model        |
| `train_questiongen.py`   | Fine-tunes on exam-style question data |
| `inference.py`           | Unified CLI interface for all tasks    |
| `models/transformer.py`  | Core model architecture (custom)       |
| `utils/tokenizer.py`     | Byte-pair tokenizer logic              |

---

## 🧪 How to Run

### 🔹 Train Language Model
```bash
python training/train_lm.py
```

### 🔹 Train True/False Classifier
```bash
python training/train_truefalse.py
```

### 🔹 Train Question Generator
```bash
python training/train_questiongen.py
```

### 🔹 Inference (all modes)
```bash
python inference.py
```

---

## ✨ Unique Contributions

- Fine-tuned a language model specifically to **generate exam questions** following the **style and rigor of CompSci 689**.
- Developed a modular training and inference setup to evaluate each capability independently.
- Built completely from scratch for educational and architectural transparency.

---

## 🔭 Future Work

1. **Data Scaling & Evaluation**  
   Identify cleaner text corpora related to statistical machine learning (textbooks, research papers) and apply a proper **train/val/test** split to benchmark generalization.

2. **SOTA Comparison**  
   Compare the performance of this custom model with fine-tuned **LLaMA-2**, **Mistral**, or **Phi** models on similar QG and classification tasks.

---

## 📚 References

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.
- UMass CS689 Lecture Materials (Fall 2024)
- Transformer paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## 🙋‍♂️ Author

**Tanishk Gali**  
CS Graduate Student, UMass Amherst  
---

## 🛠 Tech Stack

- Python 3 · PyTorch · HuggingFace Tokenizers (customized) · Transformers (manual implementation)
