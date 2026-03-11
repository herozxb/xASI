import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 399 # what is the maximum context length for predictions?
max_iters = 20000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embedding = 384
n_head = 24
n_layer = 30
dropout = 0.2
# ------------


# 7 habits of highly effective people
with open('7_habits.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    #print(text)
print(len(text))


# here are all the unique characters that occur in this text
print(set(text))
print(list(set(text)))
print(sorted(list(set(text))))
print(len(sorted(list(set(text)))))
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(stoi['\n'])
print(encode(['A','B','C']))
print(decode(encode(['A','B','C'])))


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print(train_data[0:100])
print(text[0:100])
print(sorted(list(set(text))))


#block_size = 8
print(train_data[0:block_size+1])
print(text[0:block_size+1])


x = train_data[0:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
  context = x[0:t+1]
  target = y[t]
  
  
# data loading
#torch.manual_seed(1337)
#batch_size = 4
#block_size = 8

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')
print("input:")
print(xb.shape)
print(xb)
print("target:")
print(yb.shape)
print(yb)

for b in range(batch_size):
  for t in range(block_size):
    context = xb[ b, 0:t+1 ]
    target = yb[ b, t ]
    #print(context,"->",target)
    
    
    
    
class GateLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GateLayer, self).__init__()
        
        self.linear = nn.Linear(input_dim, n_embedding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoder_output):
        # Compute gating vector
        gating_vector = self.linear(encoder_output)
        gating_vector = self.sigmoid(gating_vector)
        # Apply gating vector to encoder output
        gated_encoder_output = encoder_output * gating_vector
        
        return gated_encoder_output



class DeepSeekMLA(nn.Module):
    """ 
    DeepSeek Multi-Head Latent Attention (MLA) 
    Replaces MultiHeadAttention + Head classes
    """
    def __init__(self, n_embd, n_head, d_c=64, d_cq=64, d_rope=32):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.d_c = d_c       # KV Compression dimension
        self.d_cq = d_cq     # Query Compression dimension
        self.d_rope = d_rope # RoPE dimension



        # KV Path: Down-projection to latent space (this is what you cache)
        self.kv_down_proj = nn.Linear(n_embd, d_c, bias=False)
        self.kv_up_proj = nn.Linear(d_c, n_head * (self.head_size + d_rope), bias=False)
        
        # Query Path: Low-rank compression for training efficiency
        self.q_down_proj = nn.Linear(n_embd, d_cq, bias=False)
        self.q_up_proj = nn.Linear(d_cq, n_head * (self.head_size + d_rope), bias=False)

        self.out_proj = nn.Linear(n_head * self.head_size, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (tril)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # 1. KV Latent Projection
        c_kv = self.kv_down_proj(x) # (B, T, d_c)
        kv_up = self.kv_up_proj(c_kv).view(B, T, self.n_head, self.head_size + self.d_rope)
        # Split into content Key and RoPE Key
        k_content, k_rope = kv_up.split([self.head_size, self.d_rope], dim=-1)
        v = k_content # In MLA, V is derived from the same latent vector

        # 2. Query Latent Projection
        c_q = self.q_down_proj(x) # (B, T, d_cq)
        q_up = self.q_up_proj(c_q).view(B, T, self.n_head, self.head_size + self.d_rope)
        q_content, q_rope = q_up.split([self.head_size, self.d_rope], dim=-1)

        # 3. Attention Calculation
        # Transpose for batch-head format: (B, nh, T, d)
        q_content = q_content.transpose(1, 2)
        k_content = k_content.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute content and RoPE attention scores separately then add
        # This is a simplified version of the decoupled RoPE
        wei = (q_content @ k_content.transpose(-2, -1) + 
               q_rope @ k_rope.transpose(-2, -1)) * (self.head_size + self.d_rope)**-0.5
        
        # Masking and Softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 4. Final Aggregation
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.ReLU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Update the Block class to use the new MLA module
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Replace sa with MLA
        self.sa = DeepSeekMLA(n_embd, n_head) 
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class GPT2(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_into_embedding = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        
        #Input: (∗), IntTensor or LongTensor of arbitrary shape containing the indices to extract
        #Output: (∗,H), where * is the input shape and H=embedding_dim
        
        
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head=n_head) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embedding) # final layer norm
        #self.ffwd = FeedFoward(n_embedding)
        #self.sa_head = MultiHeadAttention(4,n_embedding//4)
        self.linear_head = nn.Linear(n_embedding, vocab_size)

    def forward(self, id_number_of_vector_x, targets=None):#target (B,T)
        #B,T =id_number_of_vector_x.shape
        #tok_embed = self.token_into_embedding(id_number_of_vector_x) #(B,T,C) (batch,Time,Channel)
        #logits = self.lm_head(tok_embed) #(B,T,vocab_size)

        B, T = id_number_of_vector_x.shape

        # id_number_of_vector_x and targets are both (B,T) tensor of integers
        #tok_emb = self.token_into_embedding(id_number_of_vector_x) # (B,T,C)
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        #x = tok_emb + pos_emb # (B,T,C) + pos_emb across the batch
        #x = self.blocks(x) # (B,T,C)
        #x = self.ln_f(x) # (B,T,C)
        
        #x = self.sa_head(x) # (B,T,vocab_size)
        #logits = self.lm_head(x) # (B,T,vocab_size)

        # id_number_of_vector_x and targets are both (B,T) tensor of integers
        tok_emb = self.token_into_embedding(id_number_of_vector_x) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        #x = tok_emb + pos_emb # (B,T,C)
        #x = self.blocks(x) # (B,T,C)
        #x = self.ln_f(x) # (B,T,C)
        #x = self.sa_head(x) # (B,T,vocab_size)
        #x = self.ffwd(x)
        #logits = self.lm_head(x) # (B,T,vocab_size)

        x = tok_emb + pos_emb # (B,T,C) + pos_emb across the batch
        x = self.blocks(x) # (B,T,C)
        x = self.layer_norm(x) # (B,T,C)
        
        #x = self.sa_head(x) # (B,T,vocab_size)
        logits = self.linear_head(x) # (B,T,vocab_size)

        if targets == None:
          loss = None
        else:
          B, T, C = logits.shape # logit(p) = ln( p / ( 1 - p ) )
          logits = logits.view(B*T,C)
          targets = targets.view(B*T)
          loss = F.cross_entropy(logits,targets) # H( P, Q ) = -0.9 * log( 0.8 ) - 0.1 * log( 0.2 ) = 0.311, the lower the better matching
        return logits, loss

    def generate(self, id_number_of_vector_x, max_new_tokens):
        # id_number_of_vector_x is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop id_number_of_vector_x to the last block_size tokens
            id_number_of_vector_x_cut = id_number_of_vector_x[:, -block_size:]
            # get the predictions
            logits, loss = self(id_number_of_vector_x_cut)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            id_number_of_vector_x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            id_number_of_vector_x = torch.cat((id_number_of_vector_x, id_number_of_vector_x_next), dim=1) # (B, T+1)
        return id_number_of_vector_x


#model = GPT2()
#m = model.to(device)
# print the number of parameters in the model
#print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#logits, loss = m(xb,yb)
#print(logits.shape,loss) # -ln(1/65) = 4.174387269896



model = GPT2()
#model.load_state_dict(torch.load("./GPT2_Shakespeare"))
model.load_state_dict(torch.load("./GPT2_rich_dad_poor_dad_Fine-tuning_with_Custom_Datasets"))
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#id_number_of_vector_x = torch.zeros((1,1), dtype=torch.long)
#print(id_number_of_vector_x)
#print(torch.tensor([[32]]).to(device))
#print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=10)[0].tolist()))

#print(decode(m.generate(torch.tensor([[32]]).to(device), max_new_tokens=388)[0].tolist()))  
  
text = "How to get rich fast"
context = torch.tensor( [ encode(text) ], dtype=torch.long, device=device )
print(decode(m.generate( context, max_new_tokens=1000 )[0].tolist()))  
