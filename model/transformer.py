import math
import numpy as np
import torch
import torch.nn as nn

from config import load_config

CONFIG = load_config()

# Token Embedding Layer
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        """
            1. What I used - too small
                - self.weights = np.random.randn(vocab_size, d_model) * 0.01 -> stats: mean: -0.0000, std: 0.0100
            
            2. Xavier / Glorot Initialization - general use, stable gradients:
                - self.weights = np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model) -> stats: mean: -0.0000, std: 0.0442

            3. Standard Normal - LLMs, emiprically proven:
                - self.weights = np.random.normal(loc=0.0, scale=0.02, size=(vocab_size, d_model))
        """
        self.weights = np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model) 

        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    
    def positional_encoding(self, pos_index, d_model_index):
        
        # pair index -> dim 2i and 2i+1
        i = d_model_index // 2 
        exponent = (2 * i) / self.d_model

        if d_model_index % 2 == 0:
            result = math.sin(pos_index / 10_000**exponent)
        else:
            result = math.cos(pos_index / 10_000**exponent)

        return result
    

    def build_positional_encoding(self, seq_len):
        """
            return -> shape: (seq_len, d_model)
        """
        pe = np.zeros((seq_len, self.d_model))

        for pos in range(seq_len):
            for dim in range(self.d_model):
                pe[pos, dim] = self.positional_encoding(pos, dim)

        return pe


    def softmax(self, vector_input):
        # Shift values so the max becomes 0

        shifted = vector_input - np.max(vector_input)
        exp_values = np.exp(shifted)

        return exp_values / np.sum(exp_values)


    def self_attention(self, embedding):
        x = torch.tensor(embedding, dtype=torch.float32)
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scale = math.sqrt(self.d_model)
        score = torch.matmul(Q, K.transpose(-2, -1)) / scale
        weights = torch.softmax(score, dim=-1)
        context = torch.matmul(weights, V)

        return Q, K, V, weights, context


    def forward(self, token_ids):
        token_embeddings = self.weights[token_ids]
        seq_len = len(token_ids)
        pe = self.build_positional_encoding(seq_len)
        return token_embeddings + pe


if __name__ == "__main__":
    vocab = CONFIG["input_ids"] # (32, 000)
    tokens = CONFIG["tokens"] # (32, 000)
    vocab_size = CONFIG["vocab_size"] # (32, 000)
    d_model = CONFIG["d_model"] # (512)

    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    input_ids = np.array(vocab, dtype=int)
    output_embedding = embedding_layer.forward(input_ids)

    print(f"Embedding size: {vocab_size} * {d_model}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output embeddings shape: {output_embedding.shape}")
    print(f"Weight stats — mean: {embedding_layer.weights.mean():.4f}, std: {embedding_layer.weights.std():.4f}")


    # Tests
    sample_tokens = tokens[:5]
    sample_ids = np.array(vocab[:5])

    print("\nSamples:")
    print(f"Sample Tokens: {sample_tokens}")
    print(f"Sample IDs: {sample_ids}")


    embedding_output = embedding_layer.forward(sample_ids)
    print(f"\nFirst {5} token embeddings ({embedding_output.shape}):")
    for tid, vec in zip(sample_tokens, embedding_output):
        print(f"  token {tid:>4} → [{', '.join(f'{v:.4f}' for v in vec[:6])}...]")


    # ---- Positional Encoding ----
    token_emb = embedding_layer.weights[sample_ids]
    pe_matrix = embedding_layer.build_positional_encoding(5)
    combined = token_emb + pe_matrix  

    dims_to_show = 6
    print("\nHow PE shifts the token embedding (first 6 dims):")
    print(f"{'':>16} | " + "  ".join(f"dim{d:<5}" for d in range(dims_to_show)))
    print("-" * (18 + dims_to_show * 9))

    for idx, (tid, token) in enumerate(zip(sample_ids, sample_tokens)):
        t = token_emb[idx]
        p = pe_matrix[idx]
        c = combined[idx]
        print(f"  {token:>12}  token | [{', '.join(f'{v:>6.4f}' for v in t[:dims_to_show])}...]")
        print(f"  {'':>12}     PE | [{', '.join(f'{v:>6.4f}' for v in p[:dims_to_show])}...]")
        print(f"  {'':>12}  total | [{', '.join(f'{v:>6.4f}' for v in c[:dims_to_show])}...]")
        print()

    
    # ---- Self-Attention ----
    Q, K, V, attention_weights, context = embedding_layer.self_attention(embedding_output) 

    dims_to_show = 6

    print("\n" + "=" * 70)
    print("SELF-ATTENTION: BEFORE vs AFTER")
    print("=" * 70)

    # ---- BEFORE: the embedding going INTO attention ----
    print("\n[BEFORE] Embedding input to self-attention (token + positional encoding)")
    print(f"Shape: {embedding_output.shape}  (seq_len={len(sample_tokens)}, d_model={d_model})")
    print(f"\n{'':>16} | " + "  ".join(f"dim{d:<5}" for d in range(dims_to_show)))
    print("-" * (18 + dims_to_show * 9))
    for token, vec in zip(sample_tokens, embedding_output):
        print(f"  {token:>12}  | [{', '.join(f'{v:>6.4f}' for v in vec[:dims_to_show])}...]")

    # ---- ATTENTION WEIGHTS: what each token focuses on ----
    print("\n[ATTENTION WEIGHTS] How much each token attends to every other token")
    print(f"Shape: {attention_weights.shape}  (each row sums to 1.0)")
    attn_np = attention_weights.detach().numpy()

    # Header row (attending TO which token)
    print(f"\n{'attending →':>20} | " + "  ".join(f"{t:>8}" for t in sample_tokens))
    print("-" * (22 + len(sample_tokens) * 10))
    for i, token in enumerate(sample_tokens):
        row = attn_np[i]
        row_str = "  ".join(f"{v:>8.4f}" for v in row)
        print(f"  {token:>16}  | {row_str}   (sum={row.sum():.4f})")

    # ---- AFTER: context vectors coming OUT of attention ----
    print("\n[AFTER] Context output from self-attention")
    print(f"Shape: {context.shape}  (seq_len={len(sample_tokens)}, d_model={d_model})")
    print(f"\n{'':>16} | " + "  ".join(f"dim{d:<5}" for d in range(dims_to_show)))
    print("-" * (18 + dims_to_show * 9))
    context_np = context.detach().numpy()
    for token, vec in zip(sample_tokens, context_np):
        print(f"  {token:>12}  | [{', '.join(f'{v:>6.4f}' for v in vec[:dims_to_show])}...]")

    # ---- DELTA: how much attention changed each token's representation ----
    print("\n[DELTA] How much self-attention shifted each token's vector (first 6 dims)")
    print(f"\n{'':>16} | " + "  ".join(f"dim{d:<5}" for d in range(dims_to_show)))
    print("-" * (18 + dims_to_show * 9))
    for token, before, after in zip(sample_tokens, embedding_output, context_np):
        delta = after[:dims_to_show] - before[:dims_to_show]
        print(f"  {token:>12}  | [{', '.join(f'{v:>+6.4f}' for v in delta)}...]")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Input  shape : {embedding_output.shape}  → (seq_len, d_model)")
    print(f"  Q/K/V  shape : {Q.shape}  → projected query/key/value")
    print(f"  Weights shape: {attn_np.shape}  → token-to-token attention scores")
    print(f"  Output shape : {context.shape}  → context-aware representations")
    print(f"\n  Mean abs shift (before → after): {np.abs(context_np - embedding_output).mean():.4f}")