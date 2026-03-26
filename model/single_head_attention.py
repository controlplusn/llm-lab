import numpy as np
import math

from config import load_config
CONFIG = load_config()

class SingleHeadAttention:
    def __init__(self, vocab_size, d_model, num_heads):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model)

        # Dimensional model for Query, Key, and Value
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Weights for Q, K, and V
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(1.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(1.0 / d_model)


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


    
    def self_attention(self, embedding):
        
        query = embedding @ self.W_q
        key = embedding @ self.W_k

        scores = query @ key.T
        return scores
    
    

    def forward(self, token_ids):
        token_embeddings = self.weights[token_ids]
        seq_len = len(token_ids)
        pe = self.build_positional_encoding(seq_len)
        
        # Combined embedding matrix
        x = token_embeddings + pe
        
        attention_output = self.self_attention(x)
        return x, attention_output

    

if __name__ == "__main__":
    vocab = CONFIG["input_ids"]
    tokens = CONFIG["tokens"]
    vocab_size = CONFIG["vocab_size"]
    d_model = CONFIG["d_model"]
    num_heads = 8

    model = SingleHeadAttention(vocab_size, d_model, num_heads)

    sample_ids = np.array(vocab[:5], dtype=int)
    sample_tokens = tokens[:5]

    combined_vectors, attention_scores = model.forward(sample_ids)

    print(f"Input shape (IDs): {sample_ids.shape}")
    print(f"Combined Embeddings shape: {combined_vectors.shape}")
    print(f"Attention Scores shape: {attention_scores.shape}")

    print(f"\nFirst 5 token embeddings (Showing first 6 dims of {d_model}):")
    for tid, vec in zip(sample_tokens, combined_vectors):
        # This no longer crashes because combined_vectors has d_model (128) columns
        print(f"  token {tid:>6} → [{', '.join(f'{v:>7.4f}' for v in vec[:6])}...]")

    print("\nSelf-Attention Raw Scores (Correlation Matrix):")
    header = f"{'':>12} |" + "".join(f"{t:>12}" for t in sample_tokens)
    print(header)
    print("-" * len(header))

    for i, row_token in enumerate(sample_tokens):
        row_str = f"{row_token:>12} |"
        for val in attention_scores[i]:
            row_str += f"{val:12.4f}"
        print(row_str)

    print("\nInterpretation:")
    print("- Higher (more positive) values indicate stronger 'affinity' between tokens.")
    print("- Diagonal values show how much a token focuses on itself.")
    print("- Since there is no Softmax, these values can be any real number.")