import json
from pathlib import Path

base_path = Path(__file__).parent
print(base_path)

load_vocab = base_path / "vocab.json"

with open(load_vocab, "r", encoding="utf-8") as f:
    vocab = json.load(f)


print("Vocabulary dictionary size:")
print(len(vocab))
