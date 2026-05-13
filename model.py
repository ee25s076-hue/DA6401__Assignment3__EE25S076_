import math
import os
from pathlib import Path
from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


#  Scaled Dot-Product Attention

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_weights = torch.softmax(scores, dim=-1)

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask, 0.0)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


#  Mask helpers

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    batch_size, tgt_len = tgt.shape
    device = tgt.device

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)      # [B, 1, 1, T]
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)                                # [1, 1, T, T]

    return pad_mask | causal_mask                              # [B, 1, T, T]


#  Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, D] -> [B, H, S, d_k]
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, S, d_k] -> [B, S, D]
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_weights

        attn_output = self.dropout(attn_output)
        concat = self._combine_heads(attn_output)
        return self.W_o(concat)

#  Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)
        return self.dropout(x)


#  Feed Forward Network

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))



#  Encoder / Decoder layers

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


#  Encoder / Decoder stacks

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else self._clone_layer(layer) for i in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    @staticmethod
    def _clone_layer(layer: EncoderLayer) -> EncoderLayer:
        import copy
        return copy.deepcopy(layer)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer if i == 0 else self._clone_layer(layer) for i in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    @staticmethod
    def _clone_layer(layer: DecoderLayer) -> DecoderLayer:
        import copy
        return copy.deepcopy(layer)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


#  Full Transformer


class _ListVocab:
    """Small vocab object used only for inference. No torchtext/datasets dependency."""
    def __init__(self, itos: List[str]) -> None:
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi.get("<unk>", 0))

    def lookup_token(self, idx: int) -> str:
        return self.itos[int(idx)]

    def get_itos(self) -> List[str]:
        return self.itos

    def get_stoi(self):
        return self.stoi


class Transformer(nn.Module):

    # Google Drive link/id of inference_bundle.pt containing model weights and vocab lists for inference mode.
    GDRIVE_CHECKPOINT_URL = "https://drive.google.com/file/d/1OtfeBTtrwwaRGHAjuUsEYW6R-5r4jSQ7/view?usp=sharing"

    DEFAULT_CHECKPOINT_PATH = "assignment3_inference_bundle.pt"
    DEFAULT_MAX_LEN = 100

    _RUNTIME_CACHE = None

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
        auto_load: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.src_vocab = None
        self.tgt_vocab = None
        self.pad_idx = 1
        self.unk_idx = 0
        self.sos_idx = 2
        self.eos_idx = 3
        self.tgt_sos_idx = 2
        self.tgt_eos_idx = 3
        self.tgt_pad_idx = 1
        self.max_len = self.DEFAULT_MAX_LEN

        called_for_inference = src_vocab_size is None or tgt_vocab_size is None
        if auto_load is None:
            auto_load = called_for_inference

        checkpoint = None
        if auto_load:
            checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
            checkpoint = self._safe_torch_load(checkpoint_path)
            if not isinstance(checkpoint, dict):
                raise RuntimeError("Downloaded checkpoint must be a dict/inference bundle.")

            config = checkpoint.get("model_config", {})
            d_model = int(config.get("d_model", d_model))
            N = int(config.get("N", config.get("layers", N)))
            num_heads = int(config.get("num_heads", config.get("heads", num_heads)))
            d_ff = int(config.get("d_ff", d_ff))
            dropout = float(config.get("dropout", dropout))

            self._load_runtime_assets_from_checkpoint(checkpoint)
            src_vocab_size = int(config.get("src_vocab_size", len(self.src_vocab)))
            tgt_vocab_size = int(config.get("tgt_vocab_size", len(self.tgt_vocab)))

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError(
                "src_vocab_size and tgt_vocab_size are required for training mode. "
                "For inference, call Transformer() and make sure GDRIVE_CHECKPOINT_URL "
                "points to inference_bundle.pt containing vocab lists."
            )

        self.src_vocab_size = int(src_vocab_size)
        self.tgt_vocab_size = int(tgt_vocab_size)
        self.d_model = int(d_model)
        self.N = int(N)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)

        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout)

        encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        self.encoder = Encoder(encoder_layer, self.N)
        self.decoder = Decoder(decoder_layer, self.N)
        self.generator = nn.Linear(self.d_model, self.tgt_vocab_size)

        self.model_config = {
            "src_vocab_size": self.src_vocab_size,
            "tgt_vocab_size": self.tgt_vocab_size,
            "d_model": self.d_model,
            "N": self.N,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        }

        self._reset_parameters()

        if checkpoint is not None:
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", None))
            if state_dict is None:
                raise RuntimeError("Inference bundle does not contain model_state_dict.")
            self.load_state_dict(state_dict, strict=True)
        elif checkpoint_path is not None:
            checkpoint = self._safe_torch_load(checkpoint_path)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _safe_torch_load(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @classmethod
    def _extract_gdrive_id(cls, url_or_id: str) -> str:
        text = str(url_or_id).strip()
        if "/d/" in text:
            return text.split("/d/", 1)[1].split("/", 1)[0]
        if "id=" in text:
            return text.split("id=", 1)[1].split("&", 1)[0]
        return text

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str]) -> str:
        candidates = []
        if checkpoint_path:
            candidates.append(Path(checkpoint_path))
        candidates.extend([
            Path(self.DEFAULT_CHECKPOINT_PATH),
            Path("checkpoints") / "inference_bundle.pt",
            Path("checkpoints") / "best_checkpoint.pt",  
            Path(__file__).resolve().parent / self.DEFAULT_CHECKPOINT_PATH,
            Path(__file__).resolve().parent / "checkpoints" / "inference_bundle.pt",
            Path.home() / ".cache" / "da6401_assignment3" / self.DEFAULT_CHECKPOINT_PATH,
        ])

        for path in candidates:
            if path.exists():
                return str(path)

        url = os.environ.get("ASSIGNMENT3_GDRIVE_CHECKPOINT", self.GDRIVE_CHECKPOINT_URL)
        if not url or "PUT_YOUR_GOOGLE_DRIVE" in url:
            raise FileNotFoundError(
                "No inference bundle found. Upload checkpoints/inference_bundle.pt to Google Drive "
                "and paste the public link/id in Transformer.GDRIVE_CHECKPOINT_URL."
            )

        cache_dir = Path.home() / ".cache" / "da6401_assignment3"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / self.DEFAULT_CHECKPOINT_PATH

        try:
            import gdown
        except ImportError as exc:
            raise ImportError("gdown is required. Add 'gdown' to requirements.txt.") from exc

        file_id = self._extract_gdrive_id(url)
        download_url = f"https://drive.google.com/uc?id={file_id}"
        result = gdown.download(download_url, str(output_path), quiet=True)

        if result is None or not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("gdown download failed or produced an empty checkpoint file.")

        return str(output_path)

    @staticmethod
    def _first_present(checkpoint: dict, names: List[str]):
        for name in names:
            if name in checkpoint:
                return checkpoint[name]
        vocab = checkpoint.get("vocab", None)
        if isinstance(vocab, dict):
            for name in names:
                if name in vocab:
                    return vocab[name]
        return None

    def _load_runtime_assets_from_checkpoint(self, checkpoint: dict) -> None:
        if self.__class__._RUNTIME_CACHE is not None:
            cache = self.__class__._RUNTIME_CACHE
            self.src_vocab = cache["src_vocab"]
            self.tgt_vocab = cache["tgt_vocab"]
            self._set_special_indices()
            return

        src_itos = self._first_present(checkpoint, ["src_itos", "src_vocab_itos", "source_itos"])
        tgt_itos = self._first_present(checkpoint, ["tgt_itos", "tgt_vocab_itos", "target_itos"])

        if src_itos is None or tgt_itos is None:
            raise RuntimeError(
                "Your downloaded file is only a weight checkpoint and does not contain vocabulary. "
                "Create checkpoints/inference_bundle.pt using make_inference_bundle.py, upload that "
                "bundle to Google Drive, and paste its link in GDRIVE_CHECKPOINT_URL. "
                "model.py must not import dataset.py on Gradescope."
            )

        self.src_vocab = _ListVocab(list(src_itos))
        self.tgt_vocab = _ListVocab(list(tgt_itos))
        self.__class__._RUNTIME_CACHE = {"src_vocab": self.src_vocab, "tgt_vocab": self.tgt_vocab}
        self._set_special_indices()

    def _set_special_indices(self) -> None:
        self.unk_idx = self._get_index(self.src_vocab, "<unk>", 0)
        self.pad_idx = self._get_index(self.src_vocab, "<pad>", 1)
        self.sos_idx = self._get_index(self.src_vocab, "<sos>", 2)
        self.eos_idx = self._get_index(self.src_vocab, "<eos>", 3)
        self.tgt_sos_idx = self._get_index(self.tgt_vocab, "<sos>", 2)
        self.tgt_eos_idx = self._get_index(self.tgt_vocab, "<eos>", 3)
        self.tgt_pad_idx = self._get_index(self.tgt_vocab, "<pad>", 1)

    @staticmethod
    def _get_index(vocab: Any, token: str, default: int) -> int:
        if vocab is None:
            return default
        if hasattr(vocab, "stoi"):
            return int(vocab.stoi.get(token, default))
        if hasattr(vocab, "get_stoi"):
            return int(vocab.get_stoi().get(token, default))
        try:
            return int(vocab[token])
        except Exception:
            return default

    @staticmethod
    def _lookup_token(vocab: Any, idx: int) -> str:
        """Safe id -> token lookup. Never crash on invalid predicted ids."""
        try:
            idx = int(idx)
        except Exception:
            return "<unk>"
        if idx < 0:
            return "<unk>"

        try:
            if hasattr(vocab, "get_itos"):
                itos = vocab.get_itos()
                return itos[idx] if idx < len(itos) else "<unk>"
            if hasattr(vocab, "itos"):
                return vocab.itos[idx] if idx < len(vocab.itos) else "<unk>"
            if hasattr(vocab, "lookup_token"):
                return vocab.lookup_token(idx)
        except Exception:
            return "<unk>"
        return str(idx)

    @staticmethod
    def _basic_tokenize(sentence: str) -> List[str]:
        import re
        return re.findall(r"\w+|[^\w\s]", sentence.lower(), flags=re.UNICODE)

    def _numericalize_src_sentence(self, sentence: str) -> torch.Tensor:
        if self.src_vocab is None:
            raise RuntimeError("Source vocabulary is not loaded. Use Transformer() for inference mode.")
        tokens = self._basic_tokenize(sentence)
        ids = [self.sos_idx]
        ids.extend(self._get_index(self.src_vocab, tok, self.unk_idx) for tok in tokens)
        ids.append(self.eos_idx)
        ids = ids[: self.max_len]
        if ids[-1] != self.eos_idx:
            ids[-1] = self.eos_idx
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    @staticmethod
    def _simple_detokenize(tokens: List[str]) -> str:
        sentence = " ".join(tokens)
        replacements = {
            " .": ".", " ,": ",", " !": "!", " ?": "?", " ;": ";", " :": ":",
            " n't": "n't", " 's": "'s", " 're": "'re", " 'm": "'m",
            " 've": "'ve", " 'll": "'ll", " 'd": "'d",
        }
        for a, b in replacements.items():
            sentence = sentence.replace(a, b)
        return sentence.strip()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    @torch.no_grad()
    def infer(self, src_sentence: str, max_len: Optional[int] = None) -> str:
        """
        Fast single-sentence German -> English inference for Gradescope.
        Uses the vocabulary stored inside inference_bundle.pt; does not import dataset.py.
        """
        self.eval()

        src_token_count = len(self._basic_tokenize(src_sentence))
        if max_len is None:
            max_len = min(32, max(8, src_token_count + 14))
        else:
            max_len = min(int(max_len), 32)

        device = next(self.parameters()).device
        src = self._numericalize_src_sentence(src_sentence).to(device)
        src_mask = make_src_mask(src, pad_idx=self.pad_idx).to(device)

        memory = self.encode(src, src_mask)
        ys = torch.full((1, 1), self.tgt_sos_idx, dtype=torch.long, device=device)

        last_id = None
        repeat_count = 0
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=self.tgt_pad_idx).to(device)
            logits = self.decode(memory, src_mask, ys, tgt_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            next_id = int(next_token.item())
            ys = torch.cat([ys, next_token], dim=1)

            if next_id == int(self.tgt_eos_idx) or next_id == int(self.tgt_pad_idx):
                break

            if next_id == last_id:
                repeat_count += 1
            else:
                repeat_count = 0
                last_id = next_id
            if repeat_count >= 3:
                break

        output_tokens: List[str] = []
        special_tokens = {"<unk>", "<pad>", "<sos>", "<bos>"}
        for idx in ys[0].tolist():
            token = self._lookup_token(self.tgt_vocab, int(idx))
            if token == "<eos>":
                break
            if token in special_tokens:
                continue
            output_tokens.append(token)

        return self._simple_detokenize(output_tokens)
