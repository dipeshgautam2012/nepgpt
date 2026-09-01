import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# Define GPT architecture here.
# The architecture is based on the GPT-2 model, which is a decoder transformer-based model for natural language processing tasks. 
# The model consists of multiple layers of self-attention and feedforward networks, with layer normalization and residual connections.

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256 # context window
    num_embed: int = 384 # embedding dimension
    num_heads: int = 6 # number of attention heads
    num_layers: int = 6 # number of transformer blocks
    dropout: float = 0.0 # dropout rate


class AttentionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head_size = config.num_embed // config.num_heads

        self.key = nn.Linear(config.num_embed, self.head_size, bias=False) #
        self.query = nn.Linear(config.num_embed, self.head_size, bias=False)
        self.value = nn.Linear(config.num_embed, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ self.value(x) # (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.num_embed, config.num_embed)
        
        # Flag to scale the projection layer that merges residual connection.
        self.proj.merges_to_residual = True
        self.dropout = nn.Dropout(config.dropout)

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, num_heads * head_size)
        out = self.proj(out) # (B, T, num_embed)
        out = self.dropout(out) # (B, T, num_embed)
        return out # (B, T, num_embed)

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.num_embed, 4 * config.num_embed), # expansion factor of 4 as described in the Attention is All You Need paper
            nn.ReLU(),
            nn.Linear(4 * config.num_embed, config.num_embed), # project back to the original embedding size
            nn.Dropout(config.dropout)
        )

        # Flag to scale the projection layer that merges residual connection.
        self.net[2].merges_to_residual = True
    
    def forward(self, x):
        return self.net(x) # (B, T, num_embed)

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.num_embed)
        self.ln2 = nn.LayerNorm(config.num_embed)
    
    def forward(self, x):
        # layers: layer normalization -> self-attention -> residual connection -> layer normalization -> feedforward -> residual connection
        x = x + self.sa(self.ln1(x)) # residual connection after self-attention # (B, T, num_embed)
        x = x + self.ffwd(self.ln2(x)) # residual connection after feedforward # (B, T, num_embed)
        return x

class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # We intentionally omit softmax in the final layer because we will use cross-entropy loss which applies softmax internally.
        # This is a common practice in PyTorch to improve efficiency.
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.num_embed)
        # The fixed sine and cosine positional encodings are replaced with learnable positional embeddings in GPT-2
        self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.num_embed) 
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.num_layers)])
        self.ln_f = nn.LayerNorm(self.config.num_embed)
        self.head = nn.Linear(self.config.num_embed, self.config.vocab_size)

        # weight sharing between token embedding and language model head
        self.token_embedding_table.weight = self.head.weight

        #initialize weights
        self.apply(self._init_weights)

    
    def _init_weights(self, module):

        # Initialize weights for linear and embedding layers; not other layrs like LayerNorm etc. 
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale down the std for residual projection layers to avoid large activations.

            if getattr(module, 'merges_to_residual', False):
                std *= (2 * self.config.num_layers) ** -0.5  # times 2: two residual additions per block
            
            nn.init.normal_(module.weight, mean=0.0, std=std) # mean=0.0, std=std as described in GPT-2 paper
            # Linear layers may have bias, so initialized to zero as described in GPT-2 paper
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,num_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,num_embed); the arange needs to on the same device as idx
        x = tok_emb + pos_emb # (B,T,num_embed)
        x = self.blocks(x) # (B,T,num_embed)
        x = self.ln_f(x) # (B,T,num_embed)
        logits = self.head(x) # (B, T, vocab_size)

        if targets is None: # target is none in case of inference, we don't need to compute the loss
            loss = None
        else: # target is not none in case of training, we need to compute the loss
            B,T,C = logits.shape # here C is vocab_size
            logits = logits.view(B*T, C) # (B*T, vocab_size)
            targets = targets.view(B*T) # (B*T,)
            loss = F.cross_entropy(logits, targets) #F.cross_entropy computes the softmax internally, so we don't need to apply softmax to logits before passing it to F.cross_entropy
            # loss is a scalar
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is input sequence; shape is (B, T), where B is batch size and T is the length of the input sequence
        for _ in range(max_new_tokens):
            # take only the last block_size tokens. This ensures input not exceeding max block size (context window)
            idx_cond = idx[:, -self.config.block_size:] 
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # get only the last time step, shape is (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence
        return idx

    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {"total": total_params, "trainable": trainable_params, "frozen": frozen_params}