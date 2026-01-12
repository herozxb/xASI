import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define parameters
num_target_words = 1000
top_num = 30
index_node = 10

# 1. Load Model & Tokenizer
model_path = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")

# 2. Filter First 10,000 Whole-Word Tokens
word_token_ids = []
vocab = tokenizer.get_vocab()

for token_str, token_id in vocab.items():
    if token_str.startswith('Ġ') or token_str.startswith(' '):  # TikToken space markers
        word_token_ids.append(token_id)
    if len(word_token_ids) >= num_target_words:
        break

word_token_tensor = torch.tensor(word_token_ids).unsqueeze(0)  # [1, 10000]

# 3. Forward Pass to get Hidden States for these 10,000 tokens
with torch.no_grad():
    # Pass word embeddings through the model layers
    outputs = model.model(word_token_tensor, output_hidden_states=True)
    final_input_h = outputs.hidden_states[-2].squeeze(0)  # [10000, 4096]

# 4. Calculate Activation for Node 0 (Top Layer Gate Proj)
top_layer = model.model.layers[-1]
node0_weights = top_layer.mlp.gate_proj.weight.data[index_node]  # [4096]

# Dot product of hidden states with neuron weights = activation level
activations = torch.matmul(final_input_h, node0_weights)  # [10000]

# 5. Get Top 10 Words by ACTIVATION level (not embedding magnitude)
top_vals, top_indices = torch.topk(activations, top_num)
top_words = [tokenizer.decode([word_token_ids[i]]) for i in top_indices.tolist()]

# 6. Generate Heatmap
heatmap_data = top_vals.unsqueeze(0).cpu().numpy()
plt.figure(figsize=(14, 4))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f",
    xticklabels=top_words, 
    yticklabels=["Node 0 Activation"],
    cmap="viridis"
)
plt.title(f"Top 30 Whole-Word Tokens for Layer 31 Node 0 (First 10k Word Subset)")
plt.xlabel("Tokens (Sorted by Activation)")
plt.tight_layout()
plt.savefig("top10_word_activations_node0.png")
plt.show()

# Print Statistics
print("\n--- Top 30 Words Activating Top Layer Node '{}' ---".format(index_node))
for i in range(top_num):
    print(f"{i+1}. '{top_words[i]}' | Activation: {top_vals[i]:.4f}")

# 7. Extract and Visualize QK Value Matrix and Key Vector Attractor
python_code = "Write a Python function that prints 'hello world' and reverse engineer the code"
inputs = tokenizer(python_code, return_tensors="pt")
input_ids = inputs['input_ids'][0]
tokens = [tokenizer.decode([t]) for t in input_ids]
initial_seq_len = len(tokens)

# Get the first layer attention mechanism
number_layer = 31
first_layer = model.model.layers[number_layer]
attention_layer = first_layer.self_attn

# Get hidden states before first layer
with torch.no_grad():
    hidden_states = model.model.embed_tokens(inputs['input_ids'])
    
    # Get the attention module's Q, K projections
    residual_states = hidden_states
    hidden_states = first_layer.input_layernorm(hidden_states)
    
    # Project to Q, K, V
    query_states = attention_layer.q_proj(hidden_states)
    key_states = attention_layer.k_proj(hidden_states)
    
    num_heads = attention_layer.num_heads
    batch_size, actual_seq_len, hidden_size_q = query_states.shape
    _, _, hidden_size_k = key_states.shape

    if hidden_size_q % num_heads != 0:
        raise ValueError(f"hidden_size_q ({hidden_size_q}) must be divisible by num_heads ({num_heads})")
    
    head_dim_q = hidden_size_q // num_heads
    num_key_value_heads = getattr(attention_layer, 'num_key_value_heads', num_heads)
    
    if hidden_size_k % num_key_value_heads != 0:
        raise ValueError(f"hidden_size_k ({hidden_size_k}) must be divisible by num_key_value_heads ({num_key_value_heads})")
    
    head_dim_k = hidden_size_k // num_key_value_heads

    seq_len = actual_seq_len
    if len(tokens) != seq_len:
        tokens = [tokenizer.decode([t]) for t in input_ids[:seq_len]]
    
    # Reshape Q: [batch, seq_len, num_heads, head_dim_q] -> [batch, num_heads, seq_len, head_dim_q]
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim_q).transpose(1, 2)
    
    # Reshape K: [batch, seq_len, num_key_value_heads, head_dim_k] -> [batch, num_key_value_heads, seq_len, head_dim_k]
    key_states_original = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim_k).transpose(1, 2)
    
    # Extract key vectors for visualization (before repeating, use original)
    key_vectors = key_states_original.mean(dim=1).squeeze(0).cpu().numpy()  # [seq_len, head_dim_k]
    
    key_states = key_states_original
    if num_key_value_heads < num_heads:
        repeat_factor = num_heads // num_key_value_heads
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
    
    if head_dim_q != head_dim_k:
        raise ValueError(f"head_dim mismatch: query has {head_dim_q}, key has {head_dim_k}")
    
    head_dim = head_dim_q  # Use query head_dim for scaling
    
    # Compute QK^T scores (raw, before softmax)
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))  # [batch, num_heads, seq_len, seq_len]
    attn_weights = attn_weights / torch.sqrt(torch.tensor(head_dim, dtype=attn_weights.dtype))
    
    # Apply causal mask (for autoregressive models)
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    causal_mask = causal_mask.to(attn_weights.device)
    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-1e-3'))  # Use -1e9 instead of -inf
    
    # Apply softmax to get attention probabilities
    attn_probs = torch.softmax(attn_weights, dim=-1)  # [batch, num_heads, seq_len, seq_len]
    
    # Get Value projection
    value_states = attention_layer.v_proj(hidden_states)
    hidden_size_v = value_states.shape[-1]
    head_dim_v = hidden_size_v // num_key_value_heads
    
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim_v).transpose(1, 2)
    
    if num_key_value_heads < num_heads:
        repeat_factor = num_heads // num_key_value_heads
        value_states = value_states.repeat_interleave(repeat_factor, dim=1)
    
    # Compute attention output
    attn_output = torch.matmul(attn_probs, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim_v]
    attn_output = attn_output.view(batch_size, seq_len, num_heads * head_dim_v)  # [batch, seq_len, hidden_size]
    
    # Apply output projection (o_proj) to get final attention output
    attn_output = attention_layer.o_proj(attn_output)  # [batch, seq_len, hidden_size]
    
    # Extract attention output for each query position
    attn_output_per_query = attn_output.squeeze(0).cpu().numpy()  # [seq_len, hidden_size]
    
    # Average across heads to get a single QK matrix for visualization
    qk_matrix = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()  # [seq_len, seq_len]
    attn_probs_avg = attn_probs.mean(dim=1).squeeze(0).cpu().numpy()  # [seq_len, seq_len]

# Print QK matrix values
print("\n=== QK Value Matrix (Raw Scores Before Softmax) ===")
print(f"Matrix shape: {qk_matrix.shape}")
print(f"QK Matrix values:\n{qk_matrix}")

# Visualize the QK matrix (raw scores before softmax)
plt.figure(figsize=(12, 10))
sns.heatmap(
    qk_matrix,
    annot=True,
    fmt=".2f",
    xticklabels=tokens,  # Query tokens are on the x-axis
    yticklabels=tokens,  # Key tokens are on the y-axis
    cmap="coolwarm",
    center=0,
    cbar_kws={'label': 'QK Raw Score'}
)

plt.title("First Layer QK Value Matrix (Raw Scores Before Softmax, Averaged Across Heads)")
plt.xlabel("Query Tokens")  # x-axis represents Query tokens
plt.ylabel("Key Tokens")    # y-axis represents Key tokens
plt.tight_layout()
plt.savefig(f"qk_value_matrix_{number_layer}.png", dpi=150)
plt.show()

# Visualize Key Vector Attractor
key_similarity = np.dot(key_vectors, key_vectors.T)  # [seq_len, seq_len]
key_norms = np.linalg.norm(key_vectors, axis=1, keepdims=True)
key_cosine_sim = key_similarity / (key_norms @ key_norms.T)

plt.figure(figsize=(12, 10))
sns.heatmap(
    key_cosine_sim,
    annot=True,
    fmt=".3f",
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Cosine Similarity'}
)
plt.title("Key Vector Attractor (Cosine Similarity Between Key Vectors)")
plt.xlabel("Token i")
plt.ylabel("Token j")
plt.tight_layout()
plt.savefig(f"key_vector_attractor_{number_layer}.png", dpi=150)
plt.show()
