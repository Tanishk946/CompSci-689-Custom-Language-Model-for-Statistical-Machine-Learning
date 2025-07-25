import math
import torch
import torch.nn as nn
from models.latent_attention import LatentAggregatorBlock, LatentDistributorBlock, DynamicMultiHeadAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)  
    def forward(self, x):
        # Add positional encoding. x shape: (B, N, d_model)
        # Slice the precomputed positions to the sequence length
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = DynamicMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # Self-attention sublayer
        attn_out, _ = self.self_attn(x, x, x)           # shape: (B, N, d_model)
        x = self.norm1(x + self.dropout(attn_out))      # residual connection
        # Feed-forward sublayer
        ff_out = self.ff(x)                             # shape: (B, N, d_model)
        x = self.norm2(x + self.dropout(ff_out))        # residual connection
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, max_seq_len=512, latent_layers=2, latent_length=64):
        super().__init__()
        assert latent_layers in (0, 2), "This implementation supports 0 or 2 latent layers (for a 4+2 split)."
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
        # Determine how many regular and latent layers
        self.num_mha_layers = num_layers - latent_layers
        self.num_latent_layers = latent_layers
        # Create standard Transformer blocks for early layers
        self.mha_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout) 
            for _ in range(self.num_mha_layers)
        ])
        # Create latent attention blocks for the last layers (if any)
        if latent_layers >= 1:
            # One latent aggregator and one distributor block
            self.latent_agg = LatentAggregatorBlock(d_model, nhead, dim_feedforward, dropout, latent_length)
        if latent_layers == 2:
            self.latent_dist = LatentDistributorBlock(d_model, nhead, dim_feedforward, dropout)
        # Final layer norm and output classifier
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_features=False):
        B, N = x.size(0), x.size(1)
        # Input embedding + positional encoding
        x = self.embed(x) * math.sqrt(self.d_model)  # scale embeddings
        x = self.pos(x)
        # Apply standard Transformer layers
        for block in self.mha_blocks:
            x = block(x)
        # Apply latent attention layers (if any)
        if self.num_latent_layers >= 1:
            # Latent aggregation: get latent vectors from full sequence
            latents = self.latent_agg(x)              # shape: (B, L, d_model)
        if self.num_latent_layers == 2:
            # Latent distribution: update token sequence using latents
            x = self.latent_dist(x, latents)          # shape: (B, N, d_model)
        # Final norm 
        x = self.norm(x)
        if return_features:
            return x  # return hidden features for external use (e.g., classification head)
        # Output logits for language modeling
        return self.out(x)

class TransformerClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        # Get features from the base transformer (use return_features to get hidden states)
        hidden = self.base(x, return_features=True)   # shape: (B, N, d_model)
        # For classification, pool the sequence (e.g., use first token or average)
        pooled = hidden[:, 0, :]                      # using first token representation
        return self.classifier(pooled)