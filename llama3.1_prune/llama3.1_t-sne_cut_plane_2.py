import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# --- Load LLaMA 3.1 ---
device = torch.device("cpu")
model_name = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B" 
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Access the embedding layer
word_embeddings = model.get_input_embeddings()

# ——— Texts ———
texts = [
    "give me a python code to write 1+2",
    "python code to write 1*2",
    "python code to 100*200",
    "python code to list all input",
    "The sky is blue.",
    "Dogs are friendly.",
    "Quantum physics is hard.",
    "Eat healthy food.",
    "He loves programming.",
    "The sun is bright.",
    "She reads books.",
    "The cat sleeps.",
    "AI is the future.",
    "This is a test sentence.",
    "The sky is red.",
    "The sky is green.",
    "The sky is blue. yes great blue",
    "Help me split the bill among my friends!",
    "write a python to add two number"
]

sentence_embeddings = []
next_token_embeddings = []
next_tokens = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(text)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1] 
        
        # 1. Calculate Sentence Embedding (Mean Pooling)
        mask = inputs["attention_mask"].unsqueeze(-1)
        sent_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        sentence_embeddings.append(sent_emb.squeeze(0).cpu().numpy())

        # 2. Predict Next Token
        logits = outputs.logits[0, -1]
        predicted_id = int(logits.argmax())
        token_text = tokenizer.decode([predicted_id])
        next_tokens.append(token_text)

        # 3. Get Next Token's Embedding Vector
        # We look up the vector for the predicted ID in the embedding matrix
        token_emb = word_embeddings(torch.tensor([predicted_id]).to(device))
        next_token_embeddings.append(token_emb.squeeze(0).cpu().numpy())

# --- Joint t-SNE ---
# Stack everything so they share the same 2D coordinate system
all_embeddings = np.vstack([sentence_embeddings, next_token_embeddings])
n_sentences = len(texts)

tsne = TSNE(n_components=2, perplexity=min(5, n_sentences-1), random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Split back into sentence and token coordinates
sent_2d = embeddings_2d[:n_sentences]
tok_2d = embeddings_2d[n_sentences:]

# --- Plot ---
plt.figure(figsize=(14, 10))

# Plot Sentences (Blue Circles)
plt.scatter(sent_2d[:, 0], sent_2d[:, 1], c='blue', label='Input Sentence', s=100, alpha=0.6)

# Plot Next Tokens (Red X's)
plt.scatter(tok_2d[:, 0], tok_2d[:, 1], c='red', marker='x', label='Predicted Next Token', s=100)

for i in range(n_sentences):
    # Label the sentence
    plt.text(sent_2d[i, 0] + 0.2, sent_2d[i, 1] + 0.2, f"S: {texts[i][:20]}...", fontsize=8, color='blue')
    
    # Label the token
    plt.text(tok_2d[i, 0] + 0.2, tok_2d[i, 1] - 0.2, f"T: '{next_tokens[i]}'", fontsize=10, color='red', fontweight='bold')
    
    # Draw arrow from sentence to its predicted token
    plt.annotate('', xy=tok_2d[i], xytext=sent_2d[i],
                 arrowprops=dict(arrowstyle="->", color='gray', alpha=0.3, lw=1))

plt.title("LLaMA 3.1 t-SNE: Sentences vs. Predicted Next Tokens")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('llama_tokens_tsne.png')
plt.show()