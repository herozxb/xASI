import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune

# =====================================
# 1. Download & prepare the data
# =====================================
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "input.txt"

if not os.path.exists(DATA_FILE):
    print(f"Downloading {DATA_FILE}...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# =====================================
# 2. Hyperparameters & data loader
# =====================================
batch_size   = 8
block_size   = 512
max_iters    = 50000
eval_interval= 500
learning_rate= 3e-4
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters   = 200
n_embd       = 128
n_head       = 8
n_layer      = 50
dropout      = 0.2

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# =====================================
# 3. Define the GPT2 model
# =====================================
class GateLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        g = self.sigmoid(self.linear(x))
        return x * g

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
        self.gate = GateLayer(n_embd, head_size)
        
        
    # tokens = ["I", "see", "a", "fox", ",", "down", "on", "you"]
    # so T = 8
    
    # x.shape == (B, T, C)  # here B=1, T=8, C=n_embd
    # so
    # x[0,6,:]  # is the embedding for the 7th token "on"
    # x[0,7,:]  # is the embedding for the 8th token "you"
    
    # x = self.gate(x)         # still (1,8,C)
    # k = self.key(x)          # (1,8,head_size)
    # q = self.query(x)        # (1,8,head_size)
    # v = self.value(x)        # (1,8,head_size)
    
    # Concretely:

    # k[0,6,:] is the key vector for “on”
    # k[0,7,:] is the key vector for “you”
    # q[0,7,:] is the query vector for “you”
    
    def forward(self, x):
        B,T,C = x.shape
        x = self.gate(x)
        k = self.key(x)   # (B,T,head)
        q = self.query(x) # (B,T,head)
        wei = q @ k.transpose(-2,-1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.blocks    = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f      = nn.LayerNorm(n_embd)
        self.lm_head   = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok = self.token_emb(idx)                    # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=device))  # (T,C)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                     # (B,T,vocab)
        loss = None
        if targets is not None:
            logits = logits.view(B*T, vocab_size)
            loss   = F.cross_entropy(logits, targets.view(B*T))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =====================================
# 4. Instantiate & initial training
# =====================================
model = GPT2().to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0 or iter==max_iters-1:
        losses = estimate_loss(model)
        print(f"iter {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")


print(decode(model.generate(torch.zeros((1,1), dtype=torch.long,device=device), max_new_tokens=100)[0].tolist()))

# =====================================
# 5. Magnitude-based pruning (30%)
# =====================================
parameters_to_prune = [
    (m, 'weight')
    for m in model.modules()
    if isinstance(m, (nn.Linear,))
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
# remove pruning reparameterization
for m, name in parameters_to_prune:
    prune.remove(m, name)

# =====================================
# 6. Fine-tune the pruned model
# =====================================
finetune_iters = 50000
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

total, zeros = 0, 0
for module, _ in parameters_to_prune:
    w = module.weight.data
    total += w.numel()
    zeros += (w==0).sum().item()
print(f"Sparsity: {zeros}/{total} = {100*zeros/total:.2f}% zeros")


for iter in range(finetune_iters):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % 1000 == 0:
        print(f"[finetune] iter {iter} loss {loss.item():.4f}")


print(decode(model.generate(torch.zeros((1,1), dtype=torch.long,device=device), max_new_tokens=100)[0].tolist()))

# =====================================
# 7. Save final model
# =====================================
torch.save(model.state_dict(), "gpt2_pruned_finetuned.pt")
print("Done. Model saved as gpt2_pruned_finetuned.pt")

