"""
Default config (see config.py):
    vocab_size = 32_000
    d_model = 512
    num_heads = 8
    num_layers = 12
    ffn_dim = 2048
    max_seq_len = 1024
"""

import math

import numpy as np
import torch
import torch.nn as nn

from config import load_config

CONFIG = load_config()


def _build_sinusoidal_pe(max_seq_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(max_seq_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    i = dim // 2
    exponent = (2 * i) / d_model
    angles = pos / np.power(10_000, exponent)
    return np.where(dim % 2 == 0, np.sin(angles), np.cos(angles)).astype(np.float32)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=math.sqrt(1.0 / d_model))

        pe = _build_sinusoidal_pe(max_seq_len, d_model)
        self.register_buffer("pe", torch.from_numpy(pe), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, seq_len]
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        batch, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        token_embeddings = self.embedding(token_ids)
        return self.dropout(token_embeddings + self.pe[:seq_len])


class LayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + 1e-6) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        # attn_mask: [batch, seq_len], True = keep token
        batch, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        if attn_mask is not None:
            # Mask padded key positions
            pad_mask = ~attn_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    # Pre-LN block: x + sublayer(LayerNorm(x))

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.mha(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    # Causal language model: token IDs in, logits [batch, seq_len, vocab_size] out

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        weight_tying: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = EmbeddingLayer(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            self.lm_head.weight = self.embedding.embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm(x)
        return self.lm_head(x)


if __name__ == "__main__":
    cfg = CONFIG
    batch, seq_len = 2, 16

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        weight_tying=cfg["weight_tying"],
    )

    input_ids = torch.randint(0, cfg["vocab_size"], (batch, seq_len))
    logits = model(input_ids)

    print(f"input_ids: {tuple(input_ids.shape)}")
    print(f"logits:    {tuple(logits.shape)}")
    assert logits.shape == (batch, seq_len, cfg["vocab_size"])
    print("OK — batched forward pass")
