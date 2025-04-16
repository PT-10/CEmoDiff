import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """
    Feature Encoder based on ECAPA-TDNN with Multilayer Feature Aggregation
    """
    def __init__(self, in_channels=80, channels=512, kernel_sizes=[5, 3, 3, 3, 3], dilations=[1, 2, 3, 4, 5]):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=kernel_sizes[0], dilation=dilations[0], padding=(kernel_sizes[0]-1)//2*dilations[0])
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        
        # SE-Res2Block layers
        self.layers = nn.ModuleList()
        for i in range(1, len(kernel_sizes)):
            self.layers.append(SERes2Block(channels, kernel_size=kernel_sizes[i], dilation=dilations[i]))
        
        # Multilayer Feature Aggregation
        self.mfa = MultilayerFeatureAggregation(channels, len(kernel_sizes))
        
    def forward(self, x):
        # x: [B, F, T] - Mel-spectrogram (batch, freq, time)
        x = x.squeeze()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        layer_outputs = [x]
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        
        # Aggregate features across layers
        x = self.mfa(layer_outputs)
        return x  # [B, C, T]


class SERes2Block(nn.Module):
    """
    SE-Res2Block as used in ECAPA-TDNN
    """
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scale - 1):
            self.convs.append(nn.Conv1d(
                self.width, self.width, kernel_size=kernel_size, 
                dilation=dilation, padding=(kernel_size-1)//2*dilation
            ))
            self.bns.append(nn.BatchNorm1d(self.width))
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
        # Squeeze-Excitation
        self.se = SqueezeExcitation(channels)
        
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        splits = torch.split(x, self.width, dim=1)
        out_splits = [splits[0]]
        
        for i in range(self.scale - 1):
            if i == 0:
                out = self.convs[i](splits[i+1])
                out = self.relu(self.bns[i](out))
            else:
                out = self.convs[i](out_splits[-1] + splits[i+1])
                out = self.relu(self.bns[i](out))
            out_splits.append(out)
            
        x = torch.cat(out_splits, dim=1)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Apply SE
        x = self.se(x)
        
        # Residual connection
        x = x + residual
        x = self.relu(x)
        
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-Excitation block for channel attention
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, T]
        batch_size, channels, _ = x.size()
        
        # Squeeze
        y = self.avg_pool(x).view(batch_size, channels)
        
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1)
        
        # Scale
        return x * y


class MultilayerFeatureAggregation(nn.Module):
    """
    Multilayer Feature Aggregation module
    """
    def __init__(self, channels, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))
        self.scale = nn.Parameter(torch.ones(1))
        self.channels = channels
        self.num_layers = num_layers
        
    def forward(self, layer_outputs):
        # Normalize weights
        norm_weights = F.softmax(self.weights, dim=0)
        
        # Weighted sum of layer outputs
        out = 0
        for i, layer_out in enumerate(layer_outputs):
            out = out + norm_weights[i] * layer_out
            
        # Scale
        out = self.scale * out
        
        return out
    
class AttentiveStatisticalPooling(nn.Module):
    """
    Attentive Statistical Pooling for global speaker embedding extraction
    """
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        # x: [B, C, T]
        # Calculate attention weights
        attn_weights = self.attention(x)  # [B, C, T]
        
        # Apply attention weights
        weighted = torch.mul(x, attn_weights)  # [B, C, T]
        
        # Compute statistics
        mean = torch.sum(weighted, dim=2, keepdim=True)  # [B, C, 1]
        var = torch.sum(torch.mul(x**2, attn_weights), dim=2, keepdim=True) - mean**2  # [B, C, 1]
        std = torch.sqrt(var.clamp(min=1e-9))  # [B, C, 1]
        
        # Concatenate mean and std
        pooled = torch.cat([mean, std], dim=1)  # [B, 2C, 1]
        pooled = pooled.squeeze(-1)  # [B, 2C]
        
        return pooled


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism used in both encoders
    """
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        
        # Project queries, keys, values
        q = self.q_proj(query)  # [B, Q_len, D]
        k = self.k_proj(key)    # [B, K_len, D]
        v = self.v_proj(value)  # [B, V_len, D]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q_len, D/H]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K_len, D/H]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, V_len, D/H]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Q_len, K_len]
        
        # Apply mask if provided
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Get weighted sum
        out = torch.matmul(attn, v)  # [B, H, Q_len, D/H]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # [B, Q_len, D]
        
        # Final projection
        out = self.out_proj(out)  # [B, Q_len, D]
        
        return out


class TimbreEncoder(nn.Module):
    """
    Timbre Encoder: Captures dynamic speaker identity from mel-spectrogram
    """
    def __init__(self, 
                 mel_dim=80, 
                 channels=512, 
                 num_latent_vectors=64, 
                 latent_dim=512,
                 num_heads=8):
        super().__init__()
        
        # Feature Encoder based on ECAPA-TDNN
        self.feature_encoder = FeatureEncoder(in_channels=mel_dim, channels=channels)
        
        # Attentive Statistical Pooling for global speaker embedding
        self.asp = AttentiveStatisticalPooling(channels)
        
        # Global speaker embedding projection
        self.speaker_proj = nn.Linear(channels * 2, latent_dim)
        
        # Trainable latent vectors for cross-attention
        self.latent_vectors = nn.Parameter(torch.randn(num_latent_vectors, latent_dim))
        
        # Cross-attention mechanism
        self.cross_attention = CrossAttention(
            query_dim=latent_dim,
            key_dim=channels,
            value_dim=channels,
            num_heads=num_heads
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: [B, F, T] - Mel-spectrogram (batch, freq bins, time frames)
        
        Returns:
            global_emb: [B, D] - Global speaker embedding
            timbre_tokens: [B, N, D] - Time-varying timbre tokens
        """
        # Get encoded features from the feature encoder
        encoded_features = self.feature_encoder(mel_spectrogram)  # [B, C, T]
        
        # Get global speaker embedding using ASP
        global_emb = self.asp(encoded_features)  # [B, 2C]
        global_emb = self.speaker_proj(global_emb)  # [B, D]
        
        # Prepare features for cross-attention
        batch_size = mel_spectrogram.size(0)
        features_t = encoded_features.transpose(1, 2)  # [B, T, C]
        
        # Expand latent vectors for batch
        query = self.latent_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, D]
        
        # Apply cross-attention to get time-varying timbre tokens
        timbre_tokens = self.cross_attention(query, features_t, features_t)  # [B, N, D]
        timbre_tokens = self.norm(timbre_tokens)  # [B, N, D]
        
        return global_emb, timbre_tokens