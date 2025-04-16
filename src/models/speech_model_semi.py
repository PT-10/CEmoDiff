import torch
import torch.nn as nn
from timbre_encoder import TimbreEncoder
from semantic_encoder import SemanticEncoder


# Complete Speech Model combining both encoders
class SpeechModel(nn.Module):
    def __init__(self, mel_dim=80):
        super().__init__()
        
        # Timbre Encoder
        self.timbre_encoder = TimbreEncoder(mel_dim=mel_dim)
        
        # Semantic Encoder
        self.semantic_encoder = SemanticEncoder()
        
    def forward(self, mel_spectrogram, waveform):
        """
        Args:
            mel_spectrogram: [B, F, T] - Speaker mel-spectrogram (batch, freq bins, time frames)
            waveform: [B, 1, T_wav] - Source raw waveform
            
        Returns:
            global_speaker_emb: [B, D] - Global speaker embedding
            semantic_repr: [B, T', D] - Rich semantic representation
        """
        # Get timbre information
        global_speaker_emb, timbre_tokens = self.timbre_encoder(mel_spectrogram)
        #Timbre tokens shape: torch.Size([2, 64, 512])
        #Waveform shape: torch.Size([2,1,16000])
        # Get semantic representation with timbre integration
        semantic_repr = self.semantic_encoder(waveform, timbre_tokens)
        # (2,1,16000) 
        return global_speaker_emb, semantic_repr


# # Example usage
# if __name__ == "__main__":
#     # Create sample inputs
#     batch_size = 2
#     mel_frames = 200
#     mel_bins = 80
#     audio_samples = 16000  # 1 second at 16kHz
    
#     mel_spectrogram = torch.randn(batch_size, mel_bins, mel_frames)
#     waveform = torch.randn(batch_size, 1, audio_samples)
    
#     # Initialize model
#     model = SpeechModel(mel_dim=mel_bins)
    
#     # Forward pass
#     global_speaker_emb, semantic_repr = model(mel_spectrogram, waveform)
    
#     print(f"Global speaker embedding shape: {global_speaker_emb.shape}")
#     print(f"Semantic representation shape: {semantic_repr.shape}")
    
