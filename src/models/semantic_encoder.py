import os
import joblib
import torch
import torch.nn as nn
import numpy as np
from transformers import HubertModel
from sklearn.cluster import MiniBatchKMeans
from timbre_encoder import CrossAttention


class ConformerBlock(nn.Module):
    """
    Modified Conformer block with added cross-attention layer
    """
    def __init__(self, dim, num_heads=8, ff_dim=2048, kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Feed-forward module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Self-attention module
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        
        # Cross-attention module (NEW)
        self.cross_attn = CrossAttention(
            query_dim=dim,
            key_dim=dim,
            value_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.cross_attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.Conv1d(dim, dim*2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(dim)

    # torch.Size([2, 49, 512]) #Timbre tokens shape: torch.Size([2, 64, 512])
    def forward(self, x, timbre_tokens=None):
        """
        Args:
            x: [B, T, D] - Input sequence
            timbre_tokens: [B, N, D] - Timbre tokens from Timbre Encoder
        
        Returns:
            x: [B, T, D] - Output sequence
        """
        # Feed-forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x_t = x.transpose(0, 1)  # [T, B, D]
        x_t, _ = self.self_attn(x_t, x_t, x_t)
        x = x_t.transpose(0, 1)  # [B, T, D]
        x = residual + self.self_attn_dropout(x)
        
        # Cross-attention with timbre tokens (if provided)
        if timbre_tokens is not None:
            residual = x
            x = self.cross_attn_norm(x)
            x = self.cross_attn(x, timbre_tokens, timbre_tokens)
            x = residual + self.cross_attn_dropout(x)
        
        # Convolution module
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv_module(x)
        x = x.transpose(1, 2)  # [B, T, D]
        x = residual + x
        
        # Feed-forward 2
        x = x + 0.5 * self.ff2(x)
        
        # Final layer norm
        x = self.norm(x)
        
        return x


class SemanticEncoder(nn.Module):
    """
    Semantic Encoder: Captures linguistic content from raw waveform
    """
    def __init__(self, 
                 conformer_dim=512, 
                 hubert_out_dim=1024,
                 num_conformer_layers=6,
                 num_heads=8,
                 ff_dim=2048,
                 n_kmeans_clusters=500):
        super().__init__()
        
        # Pretrained HuBERT model
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        
        # Freeze HuBERT parameters
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        # K-Means for quantization
        self.kmeans = MiniBatchKMeans(n_clusters=n_kmeans_clusters, batch_size=2048, random_state=0, verbose=1)

        # self.kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=0)
        self.n_kmeans_clusters = n_kmeans_clusters
        self.kmeans_trained = False
        
        # Embedding for quantized tokens
        self.token_embedding = nn.Embedding(n_kmeans_clusters, conformer_dim)
        
        # Projection from HuBERT dimensions to Conformer dimensions
        self.hubert_proj = nn.Linear(hubert_out_dim, conformer_dim)
        
        
        # Conformer encoder layers
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=conformer_dim,
                num_heads=num_heads,
                ff_dim=ff_dim
            ) for _ in range(num_conformer_layers)
        ])
        
    def train_kmeans(self, features, sampling=True, save_path="/kaggle/working/kmeans_model.pkl"):
        """
        Train or load K-Means on HuBERT features.
        Args:
            features: [N, D] - HuBERT features from multiple utterances
            sampling: whether to sample a subset for training
            save_path: path to save/load the KMeans model
        """
        if os.path.exists(save_path):
            self.kmeans = joblib.load(save_path)
            self.kmeans_trained = True
            print(f"KMeans model loaded from {save_path}")
            return
    
        if sampling:
            subset_ratio = 0.4
            subset_size = int(len(features) * subset_ratio)
            indices = np.random.choice(len(features), size=subset_size, replace=False)
            subset = features[indices]
            self.kmeans.fit(subset)
        else:
            self.kmeans.fit(features)
    
        self.kmeans_trained = True
        joblib.dump(self.kmeans, save_path)
        print(f"KMeans model trained and saved to {save_path}")

        
    def quantize(self, features):
        """
        Quantize HuBERT features using K-Means
        Args:
            features: [B, T, D] - HuBERT features
        
        Returns:
            quantized: [B, T] - Quantized token indices
        """
        batch_size, seq_len, _ = features.size()
        flattened = features.reshape(-1, features.size(-1))
        
        # If K-Means not trained, use random assignments
        if not self.kmeans_trained:
            quantized = torch.randint(0, self.n_kmeans_clusters, (flattened.size(0),))
        else:
            # Use trained K-Means for quantization
            quantized = self.kmeans.predict(flattened.cpu().numpy())
            quantized = torch.from_numpy(quantized).to(features.device)
        
        # Reshape back to batch
        quantized = quantized.reshape(batch_size, seq_len)
        
        return quantized


    #Timbre tokens shape: torch.Size([2, 64, 512])
    #Waveform shape: torch.Size([2,1,16000])
        
    def forward(self, waveform, timbre_tokens=None):
        """
        Args:
            waveform: [B, 1, T] - Raw audio waveform
            timbre_tokens: [B, N, D] - Time-varying timbre tokens from Timbre Encoder
        
        Returns:
            semantic_repr: [B, T', D] - Rich semantic representation
        """
        # Extract features using HuBERT
        with torch.no_grad():
            hubert_output = self.hubert(waveform.squeeze(1)).last_hidden_state  # [B, T', D_hubert]
            # print("Hubert output shape", hubert_output.shape)
            # Hubert output shape torch.Size([2, 49, 1024])
        
        # Project HuBERT features to conformer dimensions
        hubert_features = self.hubert_proj(hubert_output)  # [B, T', D]
        #self.hubert_proj = nn.Linear(1024, 512)
        
        # Quantize features using K-Means (offline step, but included for completeness)
        quantized_tokens = self.quantize(hubert_output)  # [B, T']
        
        # Get embeddings for quantized tokens
        token_embeddings = self.token_embedding(quantized_tokens)  # [B, T', D]
        
        # Use token embeddings as input to conformer layers
        x = token_embeddings
        # print("Shape of x", x.shape) torch.Size([2, 49, 512])
        
        # Apply conformer layers
        for layer in self.conformer_layers:
            x = layer(x, timbre_tokens)  # Pass timbre tokens to each layer
        
        return x