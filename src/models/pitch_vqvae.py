import torch
import torch.nn as nn

class PitchVQVAE(nn.Module):
    """
    VQ-VAE model for pitch quantization with a single codebook.
    """
    def __init__(self, codebook_size=64, embedding_dim=128, hidden_dim=256, 
                 reconstruction_coef=1.0, commitment_coef=0.15, vq_coef=0.05):
        super(PitchVQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.reconstruction_coef = reconstruction_coef
        self.commitment_coef = commitment_coef
        self.vq_coef = vq_coef
        
        # Encoder: 1D CNN to encode pitch contour to latent space
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=3, padding=1)
        )
        
        # Codebook for vector quantization: register as buffers
        self.register_buffer('codebook', torch.randn(codebook_size, embedding_dim))
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', self.codebook.clone())
        self.ema_decay = 0.99
        
        # Decoder: 1D CNN to decode quantized vectors back to pitch
        self.decoder = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        )
    
    def encode(self, x):
        """
        Encode input pitch to latent representation.
        x shape: [batch_size, seq_len]
        """
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        z = self.encoder(x)  # [batch_size, embedding_dim, seq_len]
        return z
    
    def decode(self, z):
        """
        Decode latent representation back to pitch.
        z shape: [batch_size, embedding_dim, seq_len]
        """
        x_recon = self.decoder(z)  # [batch_size, 1, seq_len]
        return x_recon.squeeze(1)  # [batch_size, seq_len]
    
    def vector_quantize(self, z, voiced_mask=None):
        """
        Vector quantize the latent representations using the codebook.
        """
        # Permute to shape: [batch_size, seq_len, embedding_dim]
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)  # [batch_size * seq_len, embedding_dim]
        
        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z*e
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.codebook.t())
        
        # Find nearest codebook entries
        min_encoding_indices = torch.argmin(d, dim=1)  # [batch_size * seq_len]
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
        
        # Retrieve quantized vectors
        z_q = torch.matmul(min_encodings, self.codebook)  # [batch_size * seq_len, embedding_dim]
        z_q = z_q.view(z.shape)  # [batch_size, seq_len, embedding_dim]
        
        # EMA update for codebook entries (only during training)
        if self.training:
            with torch.no_grad():
                # Count usage of each codebook entry and compute the batch sum of z vectors
                encodings_sum = min_encodings.sum(0)
                encodings_batch = torch.matmul(min_encodings.t(), z_flattened)
                
                # In-place update of ema_count and ema_weight
                self.ema_count.mul_(self.ema_decay).add_(encodings_sum * (1 - self.ema_decay))
                self.ema_weight.mul_(self.ema_decay).add_(encodings_batch * (1 - self.ema_decay))
                
                # Normalize weights by count to update the codebook in place
                n = torch.sum(self.ema_count)
                normalized_count = (self.ema_count + 1e-5) / (n + self.codebook_size * 1e-5) * n
                self.codebook.copy_(self.ema_weight / normalized_count.unsqueeze(1))
                
                # Optional: Random restart for unused codebook entries
                unused = (self.ema_count < 1e-4)
                n_unused = torch.sum(unused).int().item()
                if n_unused > 0:
                    random_indices = torch.randperm(z_flattened.shape[0])[:n_unused]
                    unused_indices = torch.nonzero(unused).squeeze()
                    # Ensure indices are of the proper shape (if only one index, unsqueeze)
                    if unused_indices.dim() == 0:
                        unused_indices = unused_indices.unsqueeze(0)
                    self.codebook[unused_indices] = z_flattened[random_indices].to(self.codebook.dtype)
        
        # Use straight-through estimator: detach quantized tensor for the encoder gradient
        z_q_sg = z_q.detach()
        z_sg = z.detach()
        
        # Loss calculations:
        # - vq_loss: encourage codebook vectors to be close to encoder outputs
        # - commitment_loss: encourage encoder outputs to commit to codebook representations
        vq_loss = torch.mean((z_sg - z_q)**2) * self.vq_coef
        commitment_loss = torch.mean((z - z_q_sg)**2) * self.commitment_coef
        
        # If a voiced mask is provided, modulate the losses accordingly
        if voiced_mask is not None:
            # Make sure voiced_mask is compatible with the flattened latent representation
            voiced_mask_flat = voiced_mask.view(-1, 1).expand_as(z_flattened)
            voiced_mask_3d = voiced_mask_flat.view(z.shape)
            vq_loss = vq_loss * voiced_mask_3d.mean()
            commitment_loss = commitment_loss * voiced_mask_3d.mean()
        
        # Reshape indices for possible downstream usage
        indices = min_encoding_indices.view(z.shape[0], z.shape[1])  # [batch_size, seq_len]
        
        # Permute quantized tensor back for the decoder: [batch_size, embedding_dim, seq_len]
        z_q = z_q.permute(0, 2, 1).contiguous()
        
        return z_q, vq_loss, commitment_loss, indices
    
    def forward(self, x, voiced_mask=None):
        """
        Forward pass through the model.
        """
        # Encode the pitch
        z = self.encode(x)
        
        # Vector quantize the latent representation
        z_q, vq_loss, commitment_loss, indices = self.vector_quantize(z, voiced_mask)
        
        # Decode the quantized representation back to pitch
        x_recon = self.decode(z_q)
        
        # Compute the reconstruction loss (using voiced_mask if provided)
        if voiced_mask is not None:
            recon_loss = torch.mean((x * voiced_mask - x_recon * voiced_mask)**2) * self.reconstruction_coef
        else:
            recon_loss = torch.mean((x - x_recon)**2) * self.reconstruction_coef
        
        loss = recon_loss + vq_loss + commitment_loss
        return x_recon, indices, loss, {
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'total_loss': loss
        }