import json
from pathlib import Path

_BASE = Path(__file__).parent
_VOCAB_PATH = _BASE / "../tokenizer/tokenizer/vocab.json"
_TOKENIZER_JSON = _BASE / "../tokenizer/tokenizer/tokenizer.json"


def load_config(vocab_path: Path = _VOCAB_PATH) -> dict:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    eot_token = "<|endoftext|>"
    eot_token_id = vocab[eot_token]

    return {
        "input_ids": list(vocab.values()),
        "tokens": list(vocab.keys()),
        "vocab_size": len(vocab),
        "eot_token": eot_token,
        "eot_token_id": eot_token_id,
        "pad_token_id": eot_token_id,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 12,
        "ffn_dim": 2048,
        "max_seq_len": 1024,
        "dropout": 0.1,
        "weight_tying": True,
        "vocab_path": str(vocab_path.resolve()),
        "tokenizer_path": str(_TOKENIZER_JSON.resolve()),
        # Training defaults (override in Colab / CLI)
        "batch_size": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "warmup_steps": 500,
        "max_steps": 50_000,
        "log_every": 50,
        "save_every": 1000,
        "num_workers": 2,
        # Data paths (relative to repo root)
        "data_jsonl": "data-cleaning/output/fineweb_cleaned.jsonl",
        "data_jsonl_sample": "data-cleaning/output/cleaned_10k.jsonl",
        "token_cache": "data-cleaning/cache/train_tokens.bin",
        "checkpoint_dir": "model/checkpoints",
        # Hugging Face dataset (set your username)
        "hf_dataset": "datasets/anaoly/fineweb-cleaned-20B",
    }
