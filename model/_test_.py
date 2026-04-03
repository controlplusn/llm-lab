import math
import torch
import torch.nn as nn

embedding = 32_000
d_model = 512
seq_len =  10
batch_size = 1

Q = nn.Linear(d_model, d_model)
K = nn.Linear(d_model, d_model)
V = nn.Linear(d_model, d_model)

print("===== Simulated Input =====")
x = torch.randn(batch_size, seq_len, d_model)
print(x)

q = Q(x)
k = K(x)
v = V(x)

print("\nQuery and Key")
print(f"Q: {q}")
print(f"K: {k}")

print("\n")
print("=" *5 + (" SCORE ") + "=" *5)
scale = math.sqrt(d_model) 
score = torch.matmul(q, k.transpose(-2, -1)) / scale
print(score.shape)

# Masking
seq = score.shape[-1] # end value of vector
vector_ones = torch.ones(seq, seq)
causal_mask = torch.triu(vector_ones, diagonal=1).bool()

print("\n")
print("=" *5 + (" CAUSAL MASK ") + "=" *5)
print(causal_mask)

# Applied masking to score (-inf) -> infinity
print("\n")
print("=" *5 + (" SCORED MASK ") + "=" *5)
masked_score = score.masked_fill(causal_mask, float('-inf'))
print(masked_score)

# Applied softmax to masked values
print("\n")
print("=" *5 + (" SOFTMAX ") + "=" *5)
weights = torch.softmax(masked_score, dim=-1)
print(weights)

print("\n")
print("=" *5 + (" VALUE MATRIX ") + "=" *5)
context = torch.matmul(weights, v)
print(context)