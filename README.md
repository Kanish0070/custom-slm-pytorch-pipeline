# TinyStories GPT Training

Train a small GPT‑style language model from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset – a collection of short, simple stories perfect for training small language models on consumer hardware.

Built with PyTorch, this project implements a decoder‑only transformer with causal self‑attention, weight tying, and mixed‑precision training. The final model can be exported to `safetensors` and GGUF formats for inference with `llama.cpp`.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Overview

Training a large language model from scratch is expensive. This project shows how to train a **Small Language Model (SLM)** on a tiny dataset (TinyStories) using only an 8 GB GPU. The model learns to generate coherent children’s stories after 10,000 training iterations.

Key features:
- **GPT architecture** – 6 layers, 6 heads, 384 embedding dimensions, 128 context length.
- **Mixed precision** – Uses `bfloat16` / `float16` with gradient scaling.
- **Efficient data pipeline** – Streams tokenized data via memory‑mapped binary files.
- **Export to GGUF** – Convert trained model to `safetensors` and then to GGUF for fast inference with `llama.cpp`.
- **All in one notebook** – The original Jupyter notebook (`training_slm.ipynb`) contains the complete workflow.

## 🧠 Architecture

The model is a standard GPT‑style decoder‑only transformer:

- **Token & position embeddings** – 50,257 vocab size (GPT‑2 tokenizer), context length 128.
- **6 transformer blocks** – Each block contains:
  - Causal self‑attention (6 heads, 384 dimensions)
  - LayerNorm (pre‑normalisation)
  - MLP (4× expansion, GELU activation)
  - Residual connections & dropout
- **Weight tying** – Input embedding and output head share the same weight matrix.
- **Final layer norm** before the output head.

Training optimizations:
- **Mixed precision** (`bfloat16` on supported GPUs, else `float16`)
- **Gradient accumulation** (32 steps) to simulate larger batch sizes
- **AdamW** with weight decay (0.1) and betas (0.9, 0.95)
- **Warmup + cosine decay** learning rate schedule
- **Gradient clipping** at 0.5

## 📊 Results

After 10,000 iterations (≈23 minutes on an RTX 4060 laptop GPU), the model reduces validation loss from **~9.4 to ~3.0**. Generated stories show reasonable coherence and creativity:

> *Once upon a time there was a pumpkin. He was so rare that he could haveher at the other animals that the spider's face and it opened it. But the bird didn't want to enjoy the top...*

<img width="554" height="432" alt="image" src="https://github.com/user-attachments/assets/d248be12-ad2d-4344-8934-48ad83f48d34" />


## 🛠️ Setup & Training

### 1. Clone the repository

```bash
git clone https://github.com/Kanish0070/custom-slm-pytorch-pipeline.git
cd custom-slm-pytorch-pipeline
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Preprocess Data
```bash
python preprocess.py
```
### 4. Train the model
```bash
python preprocess.py
```
### 5. Train the model
```bash
python preprocess.py
```
### 6. Generate text from a trained model
```bash
import torch
import tiktoken
from src.model import GPT, GPTConfig

config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load("saved_models/best_model.pt", map_location="cpu")['model_state_dict'])
model.eval()

enc = tiktoken.get_encoding("gpt2")
prompt = "Once upon a time there was a little"
input_ids = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0)
output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.7, top_k=40)
print(enc.decode(output_ids[0].tolist()))
```
### Export to GGUF (for llama.cpp)

The notebook contains cells that convert the PyTorch checkpoint to safetensors and then to GGUF. Steps:

#### 1. Save the model as model.safetensors (use save_model from safetensors.torch).

#### 2. Rename tensors to match GPT‑2 naming (transformer.h.0.ln_1 etc.) and transpose weights as required by llama.cpp.

#### 3. Use the convert.py script from llama.cpp to convert the safetensors file to GGUF.

Example command (after installing llama.cpp):

```bash
python llama.cpp/convert.py model.safetensors --outfile model.gguf --outtype f16
```

