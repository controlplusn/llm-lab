import math
import numpy as np

from config import load_config

CONFIG = load_config()

# Token Embedding Layer
class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
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


    def forward(self, token_ids):
        token_embeddings = self.weights[token_ids]
        seq_len = len(token_ids)
        pe = self.build_positional_encoding(seq_len)
        return token_embeddings + pe



if __name__ == "__main__":
    vocab = CONFIG["input_ids"]
    tokens = CONFIG["tokens"]
    vocab_size = CONFIG["vocab_size"]
    d_model = CONFIG["d_model"]

    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    input_ids = np.array(vocab, dtype=int)
    output_embedding = embedding_layer.forward(input_ids)

    print(f"Embedding size: {vocab_size} * {d_model}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output embeddings shape: {output_embedding.shape}")
    print(f"Weight stats — mean: {embedding_layer.weights.mean():.4f}, std: {embedding_layer.weights.std():.4f}")

    sample_tokens = tokens[:5]
    sample_ids = np.array(vocab[:5])

    print("\nSamples:")
    print(f"Sample Tokens: {sample_tokens}")
    print(f"Sample IDs: {sample_ids}")


    sample_vectors = embedding_layer.forward(sample_ids)
    print(f"\nFirst {5} token embeddings ({sample_vectors.shape}):")
    for tid, vec in zip(sample_tokens, sample_vectors):
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