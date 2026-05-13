from collections import Counter
from typing import Optional, Iterable, List, Tuple
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            denom = max(1, self.vocab_size - 2)  
            true_dist.fill_(self.smoothing / denom)
            true_dist[:, self.pad_idx] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[target == self.pad_idx] = 0.0

        loss = -(true_dist * log_probs).sum(dim=-1)
        non_pad = target != self.pad_idx
        return loss[non_pad].mean() if non_pad.any() else loss.sum() * 0.0


def _unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        if "src" in batch and "tgt" in batch:
            return batch["src"], batch["tgt"]
        if "de" in batch and "en" in batch:
            return batch["de"], batch["en"]
    raise TypeError("Batch must be (src, tgt) or a dict containing src/tgt tensors.")


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train(is_train)
    pad_idx = getattr(loss_fn, "pad_idx", 1)

    total_loss = 0.0
    total_tokens = 0

    iterator = tqdm(data_iter, desc=f"Epoch {epoch_num} {'train' if is_train else 'eval'}", leave=False)

    for batch in iterator:
        src, tgt = _unpack_batch(batch)
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)

        if is_train:
            assert optimizer is not None, "optimizer cannot be None during training"
            optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_gold.reshape(-1))

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        ntokens = (tgt_gold != pad_idx).sum().item()
        total_loss += loss.item() * max(1, ntokens)
        total_tokens += ntokens

        avg = total_loss / max(1, total_tokens)
        iterator.set_postfix(loss=f"{avg:.4f}")

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    if end_symbol is None:
        end_symbol = 3

    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
        logits = model.decode(memory, src_mask, ys, tgt_mask)
        next_word = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ys = torch.cat([ys, next_word], dim=1)
        if src.size(0) == 1 and next_word.item() == end_symbol:
            break

    return ys


def _get_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(int(idx))
    if hasattr(vocab, "itos"):
        return vocab.itos[int(idx)]
    if hasattr(vocab, "idx_to_token"):
        return vocab.idx_to_token[int(idx)]
    if hasattr(vocab, "get_itos"):
        return vocab.get_itos()[int(idx)]
    raise AttributeError("tgt_vocab must provide lookup_token, itos, idx_to_token, or get_itos().")


def _get_index(vocab, token: str, default: int) -> int:
    if hasattr(vocab, "stoi") and token in vocab.stoi:
        return int(vocab.stoi[token])
    if hasattr(vocab, "token_to_idx") and token in vocab.token_to_idx:
        return int(vocab.token_to_idx[token])
    if hasattr(vocab, "get_stoi") and token in vocab.get_stoi():
        return int(vocab.get_stoi()[token])
    return default


def _ids_to_tokens(ids: Iterable[int], vocab, skip_special: bool = True) -> List[str]:
    specials = {"<unk>", "<pad>", "<sos>", "<eos>", "<bos>"}
    tokens = []
    for idx in ids:
        token = _get_token(vocab, int(idx))
        if token in {"<eos>", "<sep>"}:
            break
        if skip_special and token in specials:
            continue
        tokens.append(token)
    return tokens


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(predictions: List[List[str]], references: List[List[str]], max_n: int = 4) -> float:
    if len(predictions) == 0:
        return 0.0

    pred_len = sum(len(p) for p in predictions)
    ref_len = sum(len(r) for r in references)
    if pred_len == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for pred, ref in zip(predictions, references):
            pred_counts = _ngrams(pred, n)
            ref_counts = _ngrams(ref, n)
            clipped += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            total += sum(pred_counts.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(1, pred_len))
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return 100.0 * score


@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    model.eval()

    sos_idx = _get_index(tgt_vocab, "<sos>", 2)
    eos_idx = _get_index(tgt_vocab, "<eos>", 3)
    pad_idx = _get_index(tgt_vocab, "<pad>", 1)

    predictions: List[List[str]] = []
    references: List[List[str]] = []

    for batch in tqdm(test_dataloader, desc="BLEU", leave=False):
        src, tgt = _unpack_batch(batch)
        src = src.to(device)
        tgt = tgt.to(device)

        for i in range(src.size(0)):
            single_src = src[i : i + 1]
            single_tgt = tgt[i : i + 1]
            src_mask = make_src_mask(single_src, pad_idx=pad_idx)
            decoded = greedy_decode(
                model,
                single_src,
                src_mask,
                max_len=max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )

            pred_tokens = _ids_to_tokens(decoded[0].tolist(), tgt_vocab, skip_special=True)
            ref_tokens = _ids_to_tokens(single_tgt[0].tolist(), tgt_vocab, skip_special=True)
            predictions.append(pred_tokens)
            references.append(ref_tokens)

    return _corpus_bleu(predictions, references)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": getattr(model, "model_config", {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout,
        }),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint.get("epoch", 0))
