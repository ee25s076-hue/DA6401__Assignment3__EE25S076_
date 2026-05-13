from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


class Vocabulary:
    def __init__(self, min_freq: int = 2, specials: Optional[List[str]] = None) -> None:
        self.min_freq = min_freq
        self.specials = specials if specials is not None else SPECIALS
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}

    def build(self, tokenized_sentences: Iterable[List[str]]) -> None:
        counter = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens)

        self.itos = list(self.specials)
        for token, freq in sorted(counter.items()):
            if freq >= self.min_freq and token not in self.stoi and token not in self.specials:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def lookup_token(self, idx: int) -> str:
        return self.itos[int(idx)]

    def lookup_tokens(self, indices: Iterable[int]) -> List[str]:
        return [self.lookup_token(i) for i in indices]

    def lookup_indices(self, tokens: Iterable[str]) -> List[int]:
        return [self[token] for token in tokens]

    def get_stoi(self) -> Dict[str, int]:
        return self.stoi

    def get_itos(self) -> List[str]:
        return self.itos


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_len: Optional[int] = None,
        build_vocab_now: bool = True,
    ):
        self.split = self._normalize_split(split)
        self.min_freq = min_freq
        self.max_len = max_len

        self.dataset = self._load_hf_dataset(self.split)
        self.src_tokenizer = self._load_spacy_tokenizer("de")
        self.tgt_tokenizer = self._load_spacy_tokenizer("en")

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []

        if build_vocab_now and (self.src_vocab is None or self.tgt_vocab is None):
            self.build_vocab()
        if self.src_vocab is not None and self.tgt_vocab is not None:
            self.process_data()

    @staticmethod
    def _normalize_split(split: str) -> str:
        aliases = {
            "val": "validation",
            "valid": "validation",
            "dev": "validation",
        }
        return aliases.get(split, split)

    @staticmethod
    def _load_hf_dataset(split: str):
        from datasets import load_dataset
        return load_dataset("bentrevett/multi30k", split=split)

    @staticmethod
    def _load_spacy_tokenizer(lang: str):
        import spacy
        model_name = "de_core_news_sm" if lang == "de" else "en_core_web_sm"
        try:
            return spacy.load(model_name)
        except OSError:
            return spacy.blank(lang)

    @staticmethod
    def _get_text(example, key: str) -> str:
        if key in example:
            return example[key]
        if "translation" in example:
            return example["translation"][key]
        raise KeyError(f"Could not find key '{key}' in dataset example: {example.keys()}")

    def tokenize_src(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.src_tokenizer.tokenizer(text)]

    def tokenize_tgt(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.tgt_tokenizer.tokenizer(text)]

    def build_vocab(self):
        src_sentences = []
        tgt_sentences = []
        for ex in self.dataset:
            src_sentences.append(self.tokenize_src(self._get_text(ex, "de")))
            tgt_sentences.append(self.tokenize_tgt(self._get_text(ex, "en")))

        self.src_vocab = Vocabulary(min_freq=self.min_freq)
        self.tgt_vocab = Vocabulary(min_freq=self.min_freq)
        self.src_vocab.build(src_sentences)
        self.tgt_vocab.build(tgt_sentences)
        return self.src_vocab, self.tgt_vocab

    def _numericalize(self, tokens: List[str], vocab: Vocabulary) -> List[int]:
        ids = [SOS_IDX] + vocab.lookup_indices(tokens) + [EOS_IDX]
        if self.max_len is not None:
            ids = ids[: self.max_len]
            if ids[-1] != EOS_IDX:
                ids[-1] = EOS_IDX
        return ids

    def process_data(self):
        if self.src_vocab is None or self.tgt_vocab is None:
            raise ValueError("Build or pass src_vocab and tgt_vocab before process_data().")

        self.data = []
        for ex in self.dataset:
            src_tokens = self.tokenize_src(self._get_text(ex, "de"))
            tgt_tokens = self.tokenize_tgt(self._get_text(ex, "en"))
            src_ids = self._numericalize(src_tokens, self.src_vocab)
            tgt_ids = self._numericalize(tgt_tokens, self.tgt_vocab)
            self.data.append((torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)))
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def create_dataloaders(
    batch_size: int = 64,
    min_freq: int = 2,
    max_len: Optional[int] = None,
    num_workers: int = 0,
):
    train_data = Multi30kDataset("train", min_freq=min_freq, max_len=max_len)
    val_data = Multi30kDataset(
        "validation",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        min_freq=min_freq,
        max_len=max_len,
    )
    test_data = Multi30kDataset(
        "test",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        min_freq=min_freq,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, train_data.src_vocab, train_data.tgt_vocab
