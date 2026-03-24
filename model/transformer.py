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

    def forward(self, token_ids):
        return self.weights[token_ids]



if __name__ == "__main__":
    vocab = CONFIG["input_ids"]
    vocab_size = CONFIG["vocab_size"]
    d_model = CONFIG["d_model"]

    print(f"Embedding layer size: {vocab_size * d_model}")

    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    input_ids = np.array(vocab, dtype=int)
    output_embedding = embedding_layer.forward(input_ids)

    print(f"Embedding size: {vocab_size} * {d_model}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output embeddings shape: {output_embedding.shape}")
    print(f"Weight stats — mean: {embedding_layer.weights.mean():.4f}, std: {embedding_layer.weights.std():.4f}")

    num_tokens_to_show = 5
    sample_ids = np.arange(num_tokens_to_show)
    sample_vectors = embedding_layer.forward(sample_ids)
    print(f"\nFirst {num_tokens_to_show} token embeddings ({sample_vectors.shape}):")
    for tid, vec in zip(sample_ids, sample_vectors):
        print(f"  token {tid:>4d} → [{', '.join(f'{v:.4f}' for v in vec[:6])}...]")