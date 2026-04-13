"""
Config:
    - vocab_size = 32,000
    - d_model = 512
    - num_heads = 8
"""

import math
import numpy as np
import torch
import torch.nn as nn

from config import load_config

CONFIG = load_config()



# Token Embedding Layer
class EmbeddingLayer(nn.Module):
    # token IDs -> embedding vectors -> positional encoding
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
        self.weights = np.random.randn(vocab_size, d_model) * np.sqrt(1.0 / d_model)    # Since I'm doing this just for a personal project, this is ok for now

    
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



class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + 1e-6) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

            
    def forward(self, x):
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)
        Q = Q.reshape(batch, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)   # [batch x num_heads x seq_len x d_k]
        
        K = self.W_k(x)
        K = K.reshape(batch, seq_len, self.num_heads, self.d_k)
        K = K.transpose(1, 2)  

        V = self.W_v(x)
        V = V.reshape(batch, seq_len, self.num_heads, self.d_k)
        V = V.transpose(1, 2)

        scale = math.sqrt(self.d_k)
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / scale

        mask_len = attention_score.shape[-1]
        causal_mask = torch.triu(torch.ones(mask_len, mask_len), diagonal=1).bool()

        masked_score = attention_score.masked_fill(causal_mask, float('-inf'))

        weights = torch.softmax(masked_score, dim=-1)
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2)                        # [batch × seq_len × num_heads × d_k]
        context = context.reshape(batch, seq_len, self.d_model)  # [batch × seq_len × d_model]

        output = self.W_o(context)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        self.d_model = d_model
        self.ffn_dim = ffn_dim

        self.Linear0 = nn.Linear(d_model, ffn_dim)
        self.Linear1 = nn.Linear(ffn_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.Linear1(self.relu(self.Linear0(x)))
    


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, ffn_dim)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)


    def forward(self, x):
        # attention + residual + norm
        attention_output = self.mha(x)
        x = self.norm1(attention_output + x)

        # ffn + residual + norm
        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)

        return x



if __name__ == "__main__":
    input_ids = CONFIG["input_ids"]     # Sample input for embedding layer
    d_model = CONFIG["d_model"]
    vocab_size = CONFIG["vocab_size"]
    num_heads = CONFIG["num_heads"]
    ffn_dim = CONFIG["ffn_dim"]

    batch = 2
    seq_len = 10

    dummy_input = torch.randn(batch, seq_len, d_model)

    # Embedding layer test
    embedding_layer = EmbeddingLayer(vocab_size, d_model)
    output_embedding = embedding_layer.forward(input_ids)

    print("="*20)
    print("Sample Embedding matrix")
    print("="*20) 
    print(output_embedding)

    # Expected output: [seq_len x d_model]
    print(f"Embedding Shape: {output_embedding.shape}")


    # Multi-Head Attention test
    mha = MultiHeadAttention(d_model, num_heads)
    output_mha = mha.forward(dummy_input)
    
    print()
    print("="*20)
    print("Sample Multi-Head matrix")
    print("="*20) 
    print(output_mha)

    # Expected output: [batch x seq_len x d_model]
    print(f"Multi-Head Attention Shape: {output_mha.shape}")


    # Feed Forward Network test
    ffn = FeedForward(d_model, ffn_dim)
    output_ffn = ffn.forward(dummy_input)

    print()
    print("="*20)
    print("Sample FFN matrix")
    print("="*20) 
    print(output_ffn)

    # Expected output: [batch x seq_len x d_model]
    print(f"FFN Shape: {output_ffn.shape}") 


    # Layer Norm test
    norm = LayerNorm(d_model)
    output_norm = norm.forward(dummy_input)

    print()
    print("="*20)
    print("Sample Layer Norm matrix")
    print("="*20) 
    print(output_norm)

    # Expected output: [batch x seq_len x d_model]
    print(f"Layer Norm Shape: {output_norm.shape}")


    # Transformer Block test
    transformer = TransformerBlock(d_model, num_heads, ffn_dim)
    output_transformer = transformer.forward(dummy_input)

    print()
    print("="*20)
    print("Sample Transformer Block matrix")
    print("="*20) 
    print(output_transformer)

    # Expected output: [batch x seq_len x d_model]
    print(f"Multi-Head Attention Shape: {output_transformer.shape}")