import json
from pathlib import Path

base_path = Path(__file__).parent
print(base_path)

vocab_path = base_path / "../tokenizer/tokenizer/vocab.json"

def load_config(vocab_path=vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return {
        "input_ids": list(vocab.values()),
        "vocab_size": len(vocab),
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 12,
        "ffn_dim": 2048,
        "max_seq_len": 1024,
        "dropout": 0.1,
        "weight_trying": True,
    }