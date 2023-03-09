import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256 # how many chars in a block that the transformer sees, i.e. see first predict second, see first second predict third ... see block_size chars and predict the next char
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200  # the number of iters to average the loss on, for evaluation. e.g. print the loss averaged on 200 iters
num_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(42)

# ---------------- Data ------------------- #

# read data line by line
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# get uniq chars to create vocabs
chars = sorted(list(set(text)))
vocab_size = len(chars)

# map each char to a int
ctoi = {c: i for i, c in enumerate(chars)}
# int to char
itoc = {i: c for i, c in enumerate(chars)}

# encoder takes a string and encodes to list of int
encode = lambda str: [ctoi[c] for c in str]
# decoder takes a list of int and decodes to string
decode = lambda ints: ''.join([itoc[i] for i in ints])

# encode the dataset and create tensor
data = torch.tensor(encode(text), dtype=torch.long)

# train val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ---------------- Helper functions ------------------- #

# function to get a batch of data from either train or val
def get_batch(split):
    data = train_data if split == 'train' else val_data

    # sample random index for batch
    idx = torch.randint(len(data) - block_size, (batch_size, ))  # -block_size so it ends at last char of data

    # stack the batches into rows
    x = torch.stack([data[i: i+block_size] for i in idx])
    y = torch.stack([data[i+1: i+block_size+1] for i in idx])  # transformer predicts 1->2, 1,2->3, 1,2,3->4 of the full block
    x, y = x.to(device), y.to(device)
    return x, y

# function to estimate loss, averaged over number of eval_iters, e.g. average over 200 iters
@torch.no_grad()
def estimate_loss():
    out = {}

    # put model on eval mode
    model.eval()

    # eval on both train and val set
    for split in ['train', 'val']:
        # init losses to zero
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            # get a batch
            X, Y = get_batch(split)

            # forward
            logits, loss = model(X, Y)

            # record loss of this iter
            losses[i] = loss.item()
        
        # averge over the iters
        out[split] = losses.mean()

    # put model back to train mode
    model.train()

    return out

# ---------------- Model ------------------- #
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        
        # key, query, value are pure weights multiplications, i.e. no need bias
        self.key = nn.Linear(num_embed, head_size, bias=False)      # what this token has
        self.query = nn.Linear(num_embed, head_size, bias=False)    # what this token is interested in
        self.value = nn.Linear(num_embed, head_size, bias=False)    # what this token will communicate

        # create the lower left triangle matrix of ones (for aggregating a token on all of its previous tokens)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # create key and query for each token. key = what this token contains, query = what this token is looking for
        k = self.key(x)      # (B, T, C)
        q = self.query(x)    # (B, T, C)

        # compute attention scores (affinities), by dot product between key and query, then normalize
        weights = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) ==> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # fill upper right as -inf, so that a token only looks at previous tokens
        weights = F.softmax(weights, dim=-1)  # softmax each row, so that each row is normalized
        weights = self.dropout(weights)

        # weighted aggregation of the values
        v = self.value(x)
        out = weights @ v   # (B, T, T) @ (B, T, C) => (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multi head of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_embed, num_embed)  # projection layer for residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # multi head self-attention just runs each head in parallel, then concat on the channel dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ simple linear layer"""

    def __init__(self, num_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_embed, 4 * num_embed),    # 4 * according to attention is all you need paper
            nn.ReLU(),
            nn.Linear(4 * num_embed, num_embed),    # projection layer for residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ transformer block: communication followed by computation """

    def __init__(self, num_embed, num_head):
        super().__init__()

        head_size = num_embed // num_head   # each head size
        self.sa = MultiHeadAttention(num_head, head_size)
        self.feed_forward = FeedForward(num_embed)
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))              # residual connections, so x + 
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPTModel(nn.Module):

    def __init__(self):
        super().__init__()

        # create the embedding layer, bigram takes one char and predicts next char, through a table that is nxn, where n is the number of vocabs
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed)
        self.position_embedding_table = nn.Embedding(block_size, num_embed)     # also need to know the position of the tokens

        # transformer blocks
        self.blocks = nn.Sequential(*[Block(num_embed, num_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(num_embed)     # last layer norm
        self.lm_head = nn.Linear(num_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # predict the logits of next char of all vocabs by indexing the lookup table
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # reshape logits from (B, T, C) to (B * T, C) for torch, where B is batch, T is time (like sequence of chars), C is channel (vocabs)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            # reshape target from (B, T) to (B * T) for torch
            targets = targets.view(B * T)

            # calculate loss
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # takes a batch of character, generate new ones
    # idx is (B, T) array of indices, e.g. 4 batch of 8 chars
    def generate(self, idx, max_new_tokens):

        # generate +1, +2 ... + max_new_tokens chars
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_crop = idx[:, -block_size:]

            # forward pass
            logits, loss = self(idx_crop)            # (B, T, C)

            # get only the last time step (because of bigram)
            logits = logits[:, -1, :]           # (B, C)

            # softmax logits to get probabilities
            proba = F.softmax(logits, dim=1)    # (B, C)

            # sample based on probability
            idx_next = torch.multinomial(proba, num_samples=1)  # (B, 1)

            # append sampled next idx to the running sequence
            idx = torch.concat((idx, idx_next), dim=1)          # (B, T+1)

        return idx
    
# ---------------- Train ------------------- #
model = GPTModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # eval if needed
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)

    # reset grad
    optimizer.zero_grad(set_to_none=True)

    # backward pass
    loss.backward()

    # update params
    optimizer.step()

# ---------------- Test ------------------- #
# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# generate some text and save to file
open("GPT_10000_words.txt", 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
print("done")