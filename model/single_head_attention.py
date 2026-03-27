import numpy as np
import math

from config import load_config
CONFIG = load_config()

class SingleHeadAttention:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weights = np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model)

        # Dimensional model for Query, Key, and Value
        self.head_dim = d_model

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

    
    def softmax(self, scores):
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return weights


    def self_attention(self, embedding):
        
        query = embedding @ self.W_q
        key = embedding @ self.W_k
        value = embedding @ self.W_v
        
        scale = np.sqrt(self.head_dim)
        scores = (query @ key.T) / scale
        
        weights = self.softmax(scores)

        output = weights @ value

        return scores, weights, output
    

    def forward(self, token_ids):
        token_embeddings = self.weights[token_ids]
        seq_len = len(token_ids)
        pe = self.build_positional_encoding(seq_len)
        
        # Combined embedding matrix
        x = token_embeddings + pe
        
        scores, weights, output = self.self_attention(x)
        return x, scores, weights, output

    

if __name__ == "__main__":
    vocab = CONFIG["input_ids"]
    tokens = CONFIG["tokens"]
    vocab_size = CONFIG["vocab_size"]
    d_model = CONFIG["d_model"]

    model = SingleHeadAttention(vocab_size, d_model)
    sample_ids = np.array(vocab[:5], dtype=int)
    sample_tokens = tokens[:5]

    combined_vectors, raw_scores, attention_weights, attention_output = model.forward(sample_ids)

    print(f"Input shape (IDs): {sample_ids.shape}")
    print(f"Combined Embeddings shape: {combined_vectors.shape}")
    print(f"Raw scores shape: {raw_scores.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention output shape: {attention_output.shape}")

    # Softmax verification
    print("\nWeight rows sum (Softmax verification):")
    print(np.sum(attention_weights, axis=-1))

    print("\nRaw attention scores:")
    header = f"{'':>12} |" + "".join(f"{t:>12}" for t in sample_tokens)
    print(header)
    print("-" * len(header))
    for i, row_token in enumerate(sample_tokens):
        row_str = f"{row_token:>12} |"
        for val in raw_scores[i]:
            row_str += f"{val:12.4f}"
        print(row_str)

    print("\nAttention weights (after softmax):")
    print(header)
    print("-" * len(header))
    for i, row_token in enumerate(sample_tokens):
        row_str = f"{row_token:>12} |"
        for val in attention_weights[i]:
            row_str += f"{val:12.4f}"
        print(row_str)