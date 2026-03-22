import json


def load_config(vocab_path="tokenizer/tokenizer/vocab.json"):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    return {
        "vocab_size": len(vocab),
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 12,
        "ffn_dim": 2048,
        "max_seq_len": 1024,
        "dropout": 0.1,
        "weight_trying": True,
    }