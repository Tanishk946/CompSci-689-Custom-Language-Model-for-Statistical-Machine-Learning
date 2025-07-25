import torch
import torch.nn as nn
import math

class DynamicMultiHeadAttention(nn.Module):
    """
    Custom Multi-Head Attention that properly handles dynamic sequence lengths
    """
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):
        if self.batch_first:
            batch_size = query.shape[0]
            query_seq_len = query.shape[1]
            key_seq_len = key.shape[1]
            value_seq_len = value.shape[1]
            d_model = query.shape[2]
        else:
            # Convert to batch_first for internal processing
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            batch_size = query.shape[0]
            query_seq_len = query.shape[1]
            key_seq_len = key.shape[1]
            value_seq_len = value.shape[1]
            d_model = query.shape[2]
        
        # Ensure key and value have the same sequence length
        assert key_seq_len == value_seq_len, f"Key seq_len ({key_seq_len}) != Value seq_len ({value_seq_len})"
        
        # Linear projections
        Q = self.q_proj(query)  # (B, query_seq_len, d_model)
        K = self.k_proj(key)    # (B, key_seq_len, d_model)
        V = self.v_proj(value)  # (B, value_seq_len, d_model)
        
        # Reshape for multi-head attention
        # Q: (B, query_seq_len, d_model) -> (B, query_seq_len, nhead, head_dim) -> (B, nhead, query_seq_len, head_dim)
        # K: (B, key_seq_len, d_model) -> (B, key_seq_len, nhead, head_dim) -> (B, nhead, key_seq_len, head_dim)
        # V: (B, value_seq_len, d_model) -> (B, value_seq_len, nhead, head_dim) -> (B, nhead, value_seq_len, head_dim)
        Q = Q.view(batch_size, query_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, value_seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # Q: (B, nhead, query_seq_len, head_dim)
        # K: (B, nhead, key_seq_len, head_dim)
        # Scores: (B, nhead, query_seq_len, key_seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask should be (B, key_seq_len)
            # Expand to (B, 1, 1, key_seq_len) for broadcasting
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, -1e9)
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        # attn_weights: (B, nhead, query_seq_len, key_seq_len)
        # V: (B, nhead, value_seq_len, head_dim)
        # attn_output: (B, nhead, query_seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back: (B, nhead, query_seq_len, head_dim) -> (B, query_seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_seq_len, d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        if need_weights:
            avg_attn_weights = attn_weights.mean(dim=1)  # Average across heads
            if not self.batch_first:
                avg_attn_weights = avg_attn_weights.transpose(1, 2)
            return output, avg_attn_weights
        else:
            return output, None

class LatentAggregatorBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, latent_length=64):
        super().__init__()
        # Learnable latent vectors (L x d_model)
        self.latent_length = latent_length
        self.latent_vectors = nn.Parameter(torch.randn(latent_length, d_model))
        # Cross-attention: latent queries, sequence keys/values
        self.cross_attn = DynamicMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feed-forward network for latents
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        # LayerNorms and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_sequence):
        B, N, D = token_sequence.size()
        # Expand latent vectors for each batch (B, L, d_model)
        # We unsqueeze at dim=0 to get shape (1, L, D) and repeat along batch
        latent_batch = self.latent_vectors.unsqueeze(0).expand(B, self.latent_length, D).contiguous()
        
        # Cross-attention: latents attend to the token sequence
        # Query=latent_batch (B, L, D), Key=Value=token_sequence (B, N, D)
        # Output will have shape (B, L, D) - same as query
        attn_out, _ = self.cross_attn(latent_batch, token_sequence, token_sequence)  # (B, L, D)
        
        # Residual connection on latents
        latent_out = self.norm1(latent_batch + self.dropout(attn_out))
        
        # Feed-forward on latents with another residual connection
        ff_out = self.ff(latent_out)              # (B, L, D)
        latent_out = self.norm2(latent_out + self.dropout(ff_out))
        
        return latent_out  # shape (B, L, D)

class LatentDistributorBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Cross-attention: token queries, latent keys/values
        self.cross_attn = DynamicMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Feed-forward network for tokens
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        # LayerNorms and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_sequence, latents):
        # Cross-attention: tokens attend to latents
        # Query=token_sequence (B, N, D), Key=Value=latents (B, L, D)
        # Output will have shape (B, N, D) - same as query
        attn_out, _ = self.cross_attn(token_sequence, latents, latents)  # (B, N, d_model)
        
        # Residual connection on tokens
        x = self.norm1(token_sequence + self.dropout(attn_out))
        
        # Token feed-forward
        ff_out = self.ff(x)  # (B, N, d_model)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x  # shape (B, N, d_model)