"""
Data and Tokenization utilities for REGLA reproduction.

Implements:
- GPT-2 tokenizer setup
- WikiText-103 loaders with contiguous packing into fixed-length sequences
- SlimPajama streaming iterable dataset with on-the-fly tokenization and packing

Public API:
- build_tokenizer(tokenizer_name: str = "gpt2") -> transformers.PreTrainedTokenizerFast
- pack_tokens_to_fixed_chunks(token_ids: List[int] | torch.Tensor, seq_len: int, drop_last: bool = True) -> torch.Tensor
- PackedSequenceDataset: torch.utils.data.Dataset over fixed-length contiguous chunks
- get_wt103_dataloaders(seq_len: int, batch_size: int, *, tokenizer=None, num_workers: int = 2, shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]
- SlimPajamaIterableDataset: torch.utils.data.IterableDataset yielding fixed-length packed sequences
- get_slimpajama_stream_loader(seq_len: int = 2048, batch_size: int = 8, *, tokenizer=None, num_workers: int = 0, shuffle: bool = False) -> DataLoader

Notes:
- We set tokenizer.pad_token = tokenizer.eos_token for GPT-2-like tokenizers.
- Labels are next-token targets: labels[i] = input_ids[i+1], labels[-1] = -100.
- For datasets where EOS boundaries matter, we insert eos between documents before packing.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # type: ignore

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except Exception:
    AutoTokenizer = None  # type: ignore
    PreTrainedTokenizerBase = object  # type: ignore


def build_tokenizer(tokenizer_name: str = "gpt2", cache_dir: Optional[str] = None) -> PreTrainedTokenizerBase:
    """Build or load a GPT-2 tokenizer for language modeling.

    - Sets pad_token to eos_token if not defined.
    - Returns a fast tokenizer when available.
    """
    if AutoTokenizer is None:
        raise ImportError("transformers not available; please install transformers==4.41.0")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _ensure_tensor(tokens: List[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        return tokens
    return torch.tensor(tokens, dtype=torch.long)


def pack_tokens_to_fixed_chunks(token_ids: List[int] | torch.Tensor, seq_len: int, drop_last: bool = True) -> torch.Tensor:
    """Concatenate token_ids and split into contiguous fixed-length chunks of seq_len.

    Returns a tensor of shape (num_chunks, seq_len).
    If drop_last is False and leftover exists, we right-pad with eos (assumes eos id present in data).
    """
    tokens = _ensure_tensor(token_ids)
    total = tokens.numel()
    n_full = total // seq_len
    leftover = total % seq_len
    if n_full == 0 and leftover == 0:
        return tokens.new_zeros((0, seq_len))
    if leftover != 0:
        if drop_last:
            tokens = tokens[: n_full * seq_len]
        else:
            pad = seq_len - leftover
            # Use last token as pad (ideally eos)
            pad_id = int(tokens[-1].item())
            tokens = torch.cat([tokens, tokens.new_full((pad,), pad_id)])
            n_full = (tokens.numel()) // seq_len
    chunks = tokens.view(n_full, seq_len)
    return chunks


class PackedSequenceDataset(Dataset):
    """Dataset over contiguous fixed-length sequences packed from token stream.

    Each item is a dict with:
      - input_ids: (seq_len,) long
      - labels:    (seq_len,) long where labels[t] = input_ids[t+1] and labels[-1] = -100
      - reset:     bool flag indicating start of new chunk (always True for this dataset)
    """

    def __init__(
        self,
        token_ids: List[int] | torch.Tensor,
        seq_len: int,
        *,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.chunks = pack_tokens_to_fixed_chunks(token_ids, seq_len, drop_last=drop_last)
        self.n = self.chunks.size(0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = self.chunks[idx]
        labels = x.clone()
        labels[:-1] = x[1:]
        labels[-1] = -100  # ignore last prediction
        return {"input_ids": x, "labels": labels, "reset": True}


def _collate_packed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    reset = torch.tensor([b.get("reset", True) for b in batch], dtype=torch.bool)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "reset": reset}


def _tokenize_concat_texts(dataset, field: str, tokenizer: PreTrainedTokenizerBase, eos_between_docs: bool = True) -> torch.Tensor:
    """Tokenize and concatenate all texts in a split, optionally inserting eos between docs."""
    ids: List[int] = []
    eos_id = int(tokenizer.eos_token_id)
    for ex in dataset:
        text = ex[field]
        toks = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos_between_docs:
            ids.extend(toks + [eos_id])
        else:
            ids.extend(toks)
    return torch.tensor(ids, dtype=torch.long)


def get_wt103_dataloaders(
    seq_len: int,
    batch_size: int,
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    num_workers: int = 2,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders for WikiText-103 with contiguous packing.

    Returns train_loader, val_loader, test_loader.
    """
    if load_dataset is None:
        raise ImportError("datasets not available; please install datasets==2.19.0")
    tok = tokenizer or build_tokenizer("gpt2", cache_dir=cache_dir)

    dset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)
    train_tokens = _tokenize_concat_texts(dset["train"], "text", tok, eos_between_docs=True)
    val_tokens = _tokenize_concat_texts(dset["validation"], "text", tok, eos_between_docs=True)
    test_tokens = _tokenize_concat_texts(dset["test"], "text", tok, eos_between_docs=True)

    train_ds = PackedSequenceDataset(train_tokens, seq_len, drop_last=True)
    val_ds = PackedSequenceDataset(val_tokens, seq_len, drop_last=False)
    test_ds = PackedSequenceDataset(test_tokens, seq_len, drop_last=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, collate_fn=_collate_packed)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=_collate_packed)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=_collate_packed)
    return train_loader, val_loader, test_loader


class SlimPajamaIterableDataset(IterableDataset):
    """Streaming iterable dataset for SlimPajama that yields fixed-length token sequences.

    Parameters:
    - seq_len: target sequence length
    - tokenizer: tokenizer instance
    - subset_fraction: optionally use a fraction of data for quick runs (0 < fraction <= 1)
    - shuffle_files: whether to shuffle file order (datasets streaming supports shuffling with seed)
    - seed: random seed for shuffling
    - cache_dir: optional datasets cache
    - text_field: which field to read (default "text")
    """

    def __init__(
        self,
        seq_len: int = 2048,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        *,
        subset_fraction: Optional[float] = None,
        shuffle_files: bool = False,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        text_field: str = "text",
    ) -> None:
        super().__init__()
        if load_dataset is None:
            raise ImportError("datasets not available; please install datasets==2.19.0")
        self.seq_len = int(seq_len)
        self.tokenizer = tokenizer or build_tokenizer("gpt2", cache_dir=cache_dir)
        self.subset_fraction = subset_fraction
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.cache_dir = cache_dir
        self.text_field = text_field
        # Initialize streaming dataset
        self.ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True, cache_dir=cache_dir)
        if shuffle_files:
            self.ds = self.ds.shuffle(seed=seed)
        # Estimate total examples if non-streaming; otherwise unknown

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        buffer: List[int] = []
        eos_id = int(self.tokenizer.eos_token_id)
        # Control subset by limiting number of yielded sequences if fraction provided
        max_sequences: Optional[int] = None
        if self.subset_fraction is not None:
            # Heuristic: define an upper bound for sequences per epoch
            max_sequences = int(1_000_000 * max(1e-6, min(1.0, self.subset_fraction)))
        yielded = 0
        for ex in self.ds:
            text = ex.get(self.text_field, None)
            if not isinstance(text, str):
                continue
            toks = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # insert eos between documents
            buffer.extend(toks + [eos_id])
            # yield chunks
            while len(buffer) >= self.seq_len:
                seq = buffer[: self.seq_len]
                del buffer[: self.seq_len]
                input_ids = torch.tensor(seq, dtype=torch.long)
                labels = input_ids.clone()
                labels[:-1] = input_ids[1:]
                labels[-1] = -100
                yielded += 1
                yield {"input_ids": input_ids, "labels": labels, "reset": True}
                if max_sequences is not None and yielded >= max_sequences:
                    return
        # Yield last partial padded chunk for completeness
        if len(buffer) > 0:
            pad_len = self.seq_len - len(buffer)
            buffer.extend([eos_id] * pad_len)
            input_ids = torch.tensor(buffer[: self.seq_len], dtype=torch.long)
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = -100
            yield {"input_ids": input_ids, "labels": labels, "reset": True}


def get_slimpajama_stream_loader(
    seq_len: int = 2048,
    batch_size: int = 8,
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    num_workers: int = 0,
    shuffle: bool = False,
    subset_fraction: Optional[float] = None,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """Return a DataLoader over SlimPajama streaming iterable dataset.

    Note: IterableDataset does not have a defined length. Use step-based training.
    """
    it_ds = SlimPajamaIterableDataset(
        seq_len=seq_len,
        tokenizer=tokenizer,
        subset_fraction=subset_fraction,
        shuffle_files=shuffle,
        cache_dir=cache_dir,
    )
    loader = DataLoader(it_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn=_collate_packed)
    return loader


__all__ = [
    "build_tokenizer",
    "pack_tokens_to_fixed_chunks",
    "PackedSequenceDataset",
    "get_wt103_dataloaders",
    "SlimPajamaIterableDataset",
    "get_slimpajama_stream_loader",
]
