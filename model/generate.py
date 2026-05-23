import torch
from tokenizers import Tokenizer
from pathlib import Path

from transformer import Transformer
from config import load_config

CONFIG = load_config()

tokenizer_path = Path(__file__).parent.joinpath(
    "..", "tokenizer", "tokenizer", "tokenizer.json"
).resolve()


def generate(model, tokenizer, prompt, max_new_tokens=20, device="cpu"):
    model.eval()
    model.to(device)

    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    print("=" * 50)
    print(f"Prompt : '{prompt}'")
    print(f"Token IDs : {token_ids}")
    print(f"Max new tokens: {max_new_tokens}")
    print("=" * 50)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)
            next_token_id = logits[0, -1, :].argmax().item()
            generated_ids.append(next_token_id)

            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

            if input_tensor.shape[1] > model.max_seq_len:
                input_tensor = input_tensor[:, -model.max_seq_len :]

    full_ids = token_ids + generated_ids
    full_text = tokenizer.decode(full_ids)

    print(f"Generated IDs : {generated_ids}")
    print(f"Full output : '{full_text}'")

    return full_text


if __name__ == "__main__":
    cfg = CONFIG

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
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

    generate(model, tokenizer, "The model learns to", max_new_tokens=30)
