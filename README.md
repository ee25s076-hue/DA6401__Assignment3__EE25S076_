# DA6401 – Assignment 3: Transformer for Machine Translation (German → English)

> **Course:** DA6401 – Introduction to Deep Learning  
> **Student ID:** EE25S076  
> **Task:** Implement the "Attention Is All You Need" Transformer from scratch using PyTorch and train a German → English Neural Machine Translation (NMT) system on the Multi30k dataset.

---

## 🔗 Links

| Resource | URL |
|---|---|
| 📊 WandB Report | [https://wandb.ai/shivangbhargav-krsna-/da6401-transformer-q21/reports/Assignment-3-Implementing-a-Transformer-for-Machine-Translation--VmlldzoxNjg2MzE4MA?accessToken=fkf9ss1elnwkz5b7kignha79k4w91nysud066ukux0upeiyqxrw4mid86rhta164](https://wandb.ai/YOUR_WANDB_USERNAME/YOUR_PROJECT_NAME) |
| 💻 GitHub Repository | [https://github.com/ee25s076-hue/DA6401__Assignment3__EE25S076_.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME) |
| 📄 Base Paper | [Attention Is All You Need – NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) |

> ⚠️ **Replace the placeholder WandB and GitHub URLs above with your actual links.**

---

## 📁 Project Structure

```
da6401_assignment3_ee25s076_/
│
├── model.py            # Full Transformer architecture (attention, encoder, decoder, PE)
├── train.py            # Training pipeline, loss, greedy decoding, BLEU evaluation
├── dataset.py          # Multi30k dataset loader, Vocabulary, DataLoaders
├── lr_scheduler.py     # Noam learning-rate scheduler
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 📋 Assignment Overview

This project implements the landmark **Transformer architecture** ("Attention Is All You Need", Vaswani et al., 2017) entirely from scratch in PyTorch — no `torch.nn.MultiheadAttention` used. The model is trained to translate sentences from **German to English** using the **Multi30k** dataset (~29k training pairs).

---

## 🏗️ Implementation Summary

### `model.py` – Transformer Architecture

#### Scaled Dot-Product Attention
- Implements `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`.
- Supports an optional boolean mask; masked positions are filled with `-inf` before softmax and zeroed in attention weights.

#### Masking
- `make_src_mask`: padding mask for the encoder (ignores `<pad>` tokens).
- `make_tgt_mask`: combined **padding + causal (look-ahead)** mask for the decoder, preventing any position from attending to future tokens.

#### Multi-Head Attention (`MultiHeadAttention`)
- Projects Q, K, V with independent linear layers (`W_q`, `W_k`, `W_v`, `W_o`).
- Splits the model dimension across `num_heads` heads, computes scaled dot-product attention in parallel, then concatenates and projects back.
- Dropout applied to attention output.

#### Positional Encoding (`PositionalEncoding`)
- Sinusoidal encoding: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))` and `PE(pos, 2i+1) = cos(...)`.
- Registered as a **buffer** (not a trainable parameter) via `register_buffer`.

#### Point-wise Feed-Forward Network (`PositionwiseFeedForward`)
- Two-layer linear transformation: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` with ReLU and dropout.

#### Encoder & Decoder Layers
- **`EncoderLayer`**: Self-Attention → Add & Norm → FFN → Add & Norm (Post-LayerNorm).
- **`DecoderLayer`**: Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm (Post-LayerNorm).
- `nn.LayerNorm` used for all normalisations.

#### Encoder / Decoder Stacks
- `Encoder` and `Decoder` each stack N independent copies of their respective layer using `copy.deepcopy`, followed by a final `LayerNorm`.

#### Full `Transformer`
- Combines source/target token embeddings (scaled by √d_model), positional encoding, encoder, decoder, and a linear generator head.
- Xavier uniform weight initialisation.
- Supports **training mode** (pass vocab sizes) and **inference mode** (auto-loads an `inference_bundle.pt` from Google Drive via `gdown`).
- `infer(src_sentence)`: single-sentence German → English greedy decoding for Gradescope evaluation.

---

### `train.py` – Training Pipeline

#### Label Smoothing Loss (`LabelSmoothingLoss`)
- Implements ε_ls = 0.1: distributes `0.1 / (V − 2)` probability mass uniformly across non-special tokens; concentrates `0.9` on the correct token. Pads excluded from loss.

#### `run_epoch`
- Handles a full train or validation pass over a DataLoader.
- Teacher-forcing: feeds `tgt[:, :-1]` as decoder input, computes loss against `tgt[:, 1:]`.
- Gradient clipping (`max_norm=1.0`) before each optimiser step.
- Scheduler stepped every iteration (not every epoch).

#### Greedy Decoding (`greedy_decode`)
- Auto-regressive inference: appends the argmax token at each step until `<eos>` or `max_len` is reached.

#### BLEU Evaluation (`evaluate_bleu`)
- Corpus-level BLEU (1–4 grams) computed from scratch with add-1 smoothing for each n-gram precision.
- Called via `evaluate_bleu(model, test_loader, tgt_vocab, device)`.

#### Checkpoint Utilities
- `save_checkpoint` / `load_checkpoint`: persist and restore model, optimiser, and scheduler state dictionaries along with model config.

---

### `dataset.py` – Data Pipeline

#### `Vocabulary`
- Builds token→index and index→token mappings from tokenised corpora.
- Minimum frequency filter (`min_freq=2`); inserts `<unk>`, `<pad>`, `<sos>`, `<eos>` as special tokens at fixed indices 0–3.

#### `Multi30kDataset`
- Loads the [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k) dataset via HuggingFace `datasets`.
- Tokenises German with `de_core_news_sm` and English with `en_core_web_sm` from **spaCy**.
- Numericalises tokens, wraps sequences with `<sos>` / `<eos>`, and enforces an optional `max_len`.

#### `create_dataloaders`
- Returns `(train_loader, val_loader, test_loader, src_vocab, tgt_vocab)`.
- Uses `collate_fn` with `pad_sequence` for dynamic batch padding.

---

### `lr_scheduler.py` – Noam Learning Rate Schedule

#### `NoamScheduler`
- Extends `torch.optim.lr_scheduler.LRScheduler`.
- Implements `lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})`.
- Linearly warms up for `warmup_steps` steps, then decays proportionally to the inverse square root of the step number.

#### `get_lr_history`
- Utility to simulate and return the full LR history for plotting.

---

## ⚙️ Model Configuration (Base Architecture)

| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| Encoder/Decoder layers (N) | 6 |
| Attention heads | 8 |
| d_ff (FFN inner dim) | 2048 |
| Dropout | 0.1 |
| Label smoothing (ε_ls) | 0.1 |
| Warmup steps | 4000 |
| Optimiser | Adam |
| Dataset | Multi30k (De→En) |
| Normalisation | Post-LayerNorm |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

### 2. Train the Model

```python
from dataset import create_dataloaders
from model import Transformer
from train import run_epoch, evaluate_bleu, save_checkpoint
from lr_scheduler import NoamScheduler
import torch, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(batch_size=64)

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)

for epoch in range(1, 11):
    train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                           epoch_num=epoch, is_train=True, device=device)
    val_loss   = run_epoch(val_loader,   model, loss_fn, None, None,
                           epoch_num=epoch, is_train=False, device=device)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoints/epoch_{epoch}.pt")
```

### 3. Evaluate BLEU

```python
bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
print(f"Test BLEU: {bleu:.2f}")
```


---

## 📦 Requirements

```
torch
numpy
matplotlib
scikit-learn
wandb
datasets
spacy
tqdm
gdown
```

---

## 📜 References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. [PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Multi30k Dataset: [bentrevett/multi30k on HuggingFace](https://huggingface.co/datasets/bentrevett/multi30k)
- Assignment skeleton: [MiRL-IITM/da6401_assignment_3](https://github.com/MiRL-IITM/da6401_assignment_3)