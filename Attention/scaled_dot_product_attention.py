import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Computes Scaled Dot-Product Attention.
    
    Expected shapes:
    - query: (Batch, Heads, Seq_Len_Q, d_k)
    - key:   (Batch, Heads, Seq_Len_K, d_k)
    - value: (Batch, Heads, Seq_Len_V, d_v) # Usually Seq_Len_K == Seq_Len_V and d_k == d_v
    - mask:  Broadcastable to (Batch, Heads, Seq_Len_Q, Seq_Len_K)
    """
    # 1. Get the dimension of the keys (d_k) for the scaling factor
    d_k = query.size(-1)
    
    # 2. Calculate the raw dot products (Q * K^T)
    # We transpose the last two dimensions of the Key tensor: (..., Seq_Len_K, d_k) -> (..., d_k, Seq_Len_K)
    # Matrix multiplication will result in shape: (Batch, Heads, Seq_Len_Q, Seq_Len_K)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 3. Apply the Scale (This prevents the vanishing gradients)
    scores = scores / math.sqrt(d_k)
    
    # 4. Apply Mask (if provided)
    # We fill masked positions with a very large negative number so Softmax turns them to 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 5. Apply Softmax to get the attention probabilities
    # dim=-1 means we apply softmax across the last dimension (Seq_Len_K)
    # This ensures the probabilities for all keys attending to a single query sum to 1.
    p_attn = F.softmax(scores, dim=-1)
    
    # 6. Multiply by Values (P * V)
    # (Batch, Heads, Seq_Len_Q, Seq_Len_K) @ (Batch, Heads, Seq_Len_V, d_v) 
    # Results in shape: (Batch, Heads, Seq_Len_Q, d_v)
    output = torch.matmul(p_attn, value)
    
    return output, p_attn

# --- TESTING THE IMPLEMENTATION ---

# Define dimensions
batch_size = 2
num_heads = 8
seq_len = 5       # 5 words in our sentence
d_k = 64          # Dimension of each head

# Create random dummy tensors for Q, K, V
# Normally, these come from multiplying your input by learnable weights W_q, W_k, W_v
Q = torch.randn(batch_size, num_heads, seq_len, d_k)
K = torch.randn(batch_size, num_heads, seq_len, d_k)
V = torch.randn(batch_size, num_heads, seq_len, d_k)

# Run the attention block
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("--- Shapes ---")
print(f"Query shape:   {Q.shape}")
print(f"Output shape:  {output.shape}")
print(f"Weights shape: {attention_weights.shape}")

print("\n--- Attention Probabilities (Batch 0, Head 0, Query Token 0) ---")
# Let's look at how the first token attends to all 5 tokens in the sequence
print(attention_weights[0, 0, 0, :])
print(f"Sum of probabilities: {attention_weights[0, 0, 0, :].sum().item():.2f}")