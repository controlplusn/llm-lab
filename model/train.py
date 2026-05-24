"""
Train the decoder-only transformer on packed token chunks.

Usage (from repo root):
    python model/train.py --jsonl data-cleaning/output/cleaned_10k.jsonl --max-docs 10000
    python model/train.py --token-cache data-cleaning/cache/train_tokens.bin
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from checkpoint_utils import find_latest_checkpoint, load_checkpoint, save_checkpoint
from config import load_config
from dataset import PackedLMDataset, build_token_cache, causal_lm_loss
from transformer import Transformer


def get_lr(step: int, warmup_steps: int, base_lr: float, max_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: dict, args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_root = Path(__file__).resolve().parents[1]
    token_cache = repo_root / cfg["token_cache"]

    if args.build_cache or not token_cache.exists():
        jsonl = repo_root / (args.jsonl or cfg["data_jsonl_sample"])
        print(f"Building token cache from {jsonl}")
        build_token_cache(
            jsonl_path=jsonl,
            out_path=token_cache,
            tokenizer_path=cfg["tokenizer_path"],
            eot_token_id=cfg["eot_token_id"],
            max_docs=args.max_docs,
        )

    dataset = PackedLMDataset(token_cache, seq_len=cfg["max_seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = Transformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        weight_tying=cfg["weight_tying"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"],
    )

    max_steps = args.max_steps or cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    grad_clip = cfg["grad_clip"]
    checkpoint_dir = repo_root / cfg["checkpoint_dir"]
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    step = 0
    t0 = time.time()
    running_loss = 0.0

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            batch = batch.to(device)
            lr = get_lr(step, warmup_steps, cfg["learning_rate"], max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch)
                loss = causal_lm_loss(logits, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step += 1

            if step % cfg["log_every"] == 0:
                avg = running_loss / cfg["log_every"]
                elapsed = time.time() - t0
                print(
                    f"step {step:>6} | loss {avg:.4f} | lr {lr:.2e} | "
                    f"{elapsed / step:.3f}s/step"
                )
                running_loss = 0.0

            if step % cfg["save_every"] == 0:
                save_checkpoint(
                    checkpoint_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    step,
                    cfg,
                )

    save_checkpoint(checkpoint_dir / "final.pt", model, optimizer, step, cfg)
    print("Training complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train transformer LM")
    p.add_argument("--jsonl", type=str, default=None, help="Source JSONL for token cache")
    p.add_argument("--token-cache", type=str, default=None, help="Prebuilt .bin cache")
    p.add_argument("--build-cache", action="store_true", help="Force rebuild token cache")
    p.add_argument("--max-docs", type=int, default=None, help="Limit docs when building cache")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(load_config(), parse_args())
