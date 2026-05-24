
from __future__ import annotations

import re
from pathlib import Path

import torch

_CONFIG_KEYS = (
    "vocab_size",
    "d_model",
    "num_heads",
    "ffn_dim",
    "num_layers",
    "max_seq_len",
    "dropout",
    "weight_tying",
    "eot_token_id",
)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    # Return the highest step_*.pt in checkpoint_dir, else final.pt, else None
    
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    step_ckpts = list(checkpoint_dir.glob("step_*.pt"))
    if step_ckpts:
        def _step_num(path: Path) -> int:
            match = re.search(r"step_(\d+)", path.stem)
            return int(match.group(1)) if match else -1

        return max(step_ckpts, key=_step_num)

    final = checkpoint_dir / "final.pt"
    return final if final.exists() else None


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    cfg: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model": model.state_dict(),
        "config": {k: cfg[k] for k in _CONFIG_KEYS if k in cfg},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path)
    size_mb = path.stat().st_size / 1e6
    print(f"Saved checkpoint -> {path} ({size_mb:.1f} MB)")
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> int:
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    step = int(ckpt.get("step", 0))
    print(f"Loaded checkpoint -> {path} (step {step})")
    return step
