import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# ——— Load LLaMA 3.1 tokenizer and model ———

device       = torch.device("cpu")

model_name = "/Users/xibozhang/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B"  # adjust to your path

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()
print("===============1=============")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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

# ——— Embeddings and next-token predictions ———
embeddings = []
next_tokens = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(text)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden = outputs.hidden_states[-1]   # last layer hidden states
        logits = outputs.logits

        # mean pooling with attention mask
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [1, seq, 1]
        masked_hidden = hidden * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1)         # avoid div0
        sentence_embedding = sum_hidden / lengths                # [1, hidden_dim]

        embeddings.append(sentence_embedding.squeeze(0).detach().cpu().numpy())

        # next-token prediction
        next_token_logits = logits[0, -1]
        predicted_id = int(next_token_logits.argmax())
        next_tokens.append(tokenizer.decode([predicted_id]))



# ——— t-SNE to 2D ———
tsne = TSNE(n_components=2, perplexity=min(5, len(texts)-1), random_state=42)

embeddings_2d = tsne.fit_transform(np.stack(embeddings))

print("===============2=============")

# ——— Plot ———
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from matplotlib import cm

# Create unique label set and mapping
unique_labels = sorted(set(next_tokens))  # ensure consistent order

#label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
#y = np.array([label_to_index[token] for token in next_tokens])  # class indices

# ——— Encode labels for classification ———
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_index[token] for token in next_tokens])

# ——— Train KNN classifier on t-SNE 2D ———
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(embeddings_2d, y)

# ——— Generate decision boundary mesh ———
h = 0.3  # step size in the mesh

x_min, x_max = embeddings_2d[:, 0].min() - 1, embeddings_2d[:, 0].max() + 1
y_min, y_max = embeddings_2d[:, 1].min() - 1, embeddings_2d[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print("===============3=============")

# ——— Plot decision boundaries ———
plt.figure(figsize=(12, 8))

cmap = ListedColormap(cm.get_cmap("tab10").colors[:len(unique_labels)])

plt.contourf(xx, yy, Z, alpha=0.2, cmap=cmap)

print("===============4=============")

# ——— Plot points again ———
for i, (x, y_pt) in enumerate(embeddings_2d):

    label = next_tokens[i]
    color_idx = label_to_index[label]
    
    plt.scatter(x, y_pt, color=cmap(color_idx), label=label if label not in next_tokens[:i] else "")
    plt.text(x + 0.5, y_pt, texts[i], fontsize=8)

print("===============5=============")

plt.title("LLaMA 3.1 t-SNE Cut Plane with Contours: Next Token Prediction")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend(title="Predicted Next Token")
plt.grid(True)
plt.tight_layout()

# Saving the plot
plt.savefig('plot_output.png')

plt.show()


