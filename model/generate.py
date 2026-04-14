import torch
from tokenizers import Tokenizer
from pathlib import Path
 
from transformer import Transformer, EmbeddingLayer
from config import load_config
 
CONFIG = load_config()

tokenizer_path = Path(__file__).parent.joinpath("..", "tokenizer", "tokenizer", "tokenizer.json").resolve()


def generate(model, tokenizer, prompt, max_new_tokens=20, device="cpu"):
    model.eval()
    model.to(device)

    # Encoder: string -> token IDs
    encoded = tokenizer.encode(prompt)
    token_ids = encoded.ids
    input_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)

    print("=" * 50)
    print(f"Prompt : '{prompt}'")
    print(f"Token IDs : {token_ids}")
    print(f"Tokens : {[tokenizer.decode([t]) for t in token_ids]}")
    print(f"Max new tokens: {max_new_tokens}")
    print("=" * 50)
    print()

    generated_ids = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass -> logits: [1, seq_len, vocab_size]
            logits = model.forward(input_tensor)
 
            # Take logits at the last position only -> [vocab_size]
            next_token_logits = logits[0, -1, :]
 
            # Greedy decoding: pick the highest scoring token
            next_token_id = torch.argmax(next_token_logits).item()
            # next_token_str = tokenizer.decode([next_token_id])
 
            # print(f"Step {step + 1:>3} | token_id: {next_token_id:<6} | token: '{next_token_str}'")
 
            generated_ids.append(next_token_id)
 
            # Append predicted token to the running sequence
            next_tensor = torch.tensor([next_token_id], dtype=torch.long).to(device)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=0)

    
    print()

    full_ids = token_ids + generated_ids
    full_text = tokenizer.decode(full_ids)

    print(f"Generated IDs : {generated_ids}")
    print(f"Full output : '{full_text}'")
 
    return full_text


if __name__ == "__main__":
    vocab_size = CONFIG["vocab_size"]
    d_model = CONFIG["d_model"]
    num_heads = CONFIG["num_heads"]
    ffn_dim = CONFIG["ffn_dim"]
    num_layers = CONFIG.get("num_layers")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers
    )

    prompt = "The model learns to"
    generate(model, tokenizer, prompt, max_new_tokens=30)