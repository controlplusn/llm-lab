"""
Packed causal LM dataset.

Tokenizes JSONL {"text": ...} lines, inserts <|endoftext|> between documents,
and slices into fixed-length chunks for training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tqdm import tqdm


def _load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    path = Path(tokenizer_path)
    if path.exists():
        return Tokenizer.from_file(str(path))

    base = path.parent
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel

    tok = Tokenizer(BPE.from_file(str(base / "vocab.json"), str(base / "merges.txt")))
    tok.pre_tokenizer = ByteLevel()
    return tok


def build_token_cache(
    jsonl_path: str | Path,
    out_path: str | Path,
    tokenizer_path: str | Path,
    eot_token_id: int = 0,
    max_docs: int | None = None,
) -> Path:
    """
    Stream JSONL -> uint32 memmap file of concatenated token IDs (with EOT separators).
    """
    jsonl_path = Path(jsonl_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(tokenizer_path)
    tmp_path = out_path.with_suffix(".tmp.bin")

    total_tokens = 0
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(tmp_path, "wb") as f_out:
        for i, line in enumerate(tqdm(f_in, desc="Tokenizing")):
            if max_docs is not None and i >= max_docs:
                break
            row = json.loads(line)
            text = row.get("text", "")
            if not text.strip():
                continue
            ids = tokenizer.encode(text).ids
            ids.append(eot_token_id)
            arr = np.array(ids, dtype=np.uint32)
            arr.tofile(f_out)
            total_tokens += len(ids)

    if total_tokens == 0:
        raise ValueError(f"No tokens written from {jsonl_path}")

    data = np.memmap(tmp_path, dtype=np.uint32, mode="r", shape=(total_tokens,))
    out_mmap = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))
    out_mmap[:] = data
    del data
    del out_mmap
    tmp_path.unlink(missing_ok=True)

    meta = {
        "total_tokens": int(total_tokens),
        "dtype": "uint32",
        "eot_token_id": eot_token_id,
        "source": str(jsonl_path.resolve()),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {total_tokens:,} tokens -> {out_path}")
    return out_path


class PackedLMDataset(Dataset):
    # Fixed-length chunks from a prebuilt token cache

    def __init__(self, token_path: str | Path, seq_len: int):
        self.seq_len = seq_len
        path = Path(token_path)
        meta_path = path.with_suffix(".meta.json")

        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            total = meta["total_tokens"]
        else:
            total = path.stat().st_size // np.dtype(np.uint32).itemsize

        self.tokens = np.memmap(path, dtype=np.uint32, mode="r", shape=(total,))
        # Drop trailing tokens that do not fill a full chunk
        self.num_chunks = len(self.tokens) // seq_len

        if self.num_chunks == 0:
            raise ValueError(
                f"Need at least {seq_len} tokens in cache, got {len(self.tokens)}"
            )

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = np.array(self.tokens[start:end], dtype=np.int64)
        return torch.from_numpy(chunk)


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  [batch, seq_len, vocab_size]
    targets: [batch, seq_len]  (same as input_ids; predicts next token)
    """
    batch, seq_len, vocab_size = logits.shape
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size),
        targets[:, 1:].reshape(-1),
    )
