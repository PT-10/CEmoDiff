import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import SemanticPredictor, SpeakerClassifier


class MultiTaskLoss:
    """Multi-task loss for jointly training timbre and semantic encoders"""
    def __init__(self, 
                 num_speakers, 
                 contrastive_weight=1.0, 
                 semantic_weight=1.0, 
                 speaker_weight=1.0,
                 temperature=0.07):
        
        self.speaker_classifier = SpeakerClassifier(512, num_speakers).to(device)
        self.semantic_predictor = SemanticPredictor(512, 128).to(device)
        self.speaker_projector = nn.Linear(512, 128).to(device)
        
        self.contrastive_weight = contrastive_weight
        self.semantic_weight = semantic_weight
        self.speaker_weight = speaker_weight
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def get_speaker_loss(self, global_speaker_emb, speaker_ids):
        """Speaker classification loss"""
        logits = self.speaker_classifier(global_speaker_emb)
        return self.ce_loss(logits, speaker_ids)
    
    def get_contrastive_loss(self, global_speaker_emb, semantic_repr):
        
        """Contrastive loss between speaker embedding and semantic representation"""
        batch_size = global_speaker_emb.size(0)
        
        # Get mean pooled semantic representation
        semantic_mean = torch.mean(semantic_repr, dim=1)  # [B, D]
        
        # Project semantic representation
        semantic_proj = self.semantic_predictor(semantic_mean)
        speaker_proj = self.speaker_projector(global_speaker_emb)   # [B, 128]
        # semantic_proj = semantic_mean
        
        # Normalize embeddings
        # global_speaker_emb = F.normalize(global_speaker_emb, p=2, dim=1)
        speaker_proj = F.normalize(speaker_proj, p=2, dim=1)
        semantic_proj = F.normalize(semantic_proj, p=2, dim=1)
        # print(speaker_proj, semantic_proj.T)
        # Compute similarity matrix
        sim_matrix = torch.matmul(speaker_proj, semantic_proj.T) / self.temperature
        
        # Labels are the diagonal indices (same sample positive pairs)
        labels = torch.arange(batch_size).to(device)
        
        # Compute contrastive loss (both directions)
        # print(sim_matrix.shape, labels.shape)
        loss_speaker_to_semantic = self.ce_loss(sim_matrix, labels)
        loss_semantic_to_speaker = self.ce_loss(sim_matrix.T, labels)
        
        return (loss_speaker_to_semantic + loss_semantic_to_speaker) / 2
    
    def get_semantic_consistency_loss(self, semantic_repr, timbre_tokens):
        """Semantic consistency loss to ensure semantic and timbre are complementary"""
        # Mean pool semantic representation
        semantic_mean = torch.mean(semantic_repr, dim=1)  # [B, D]
        
        # Mean pool timbre tokens
        timbre_mean = torch.mean(timbre_tokens, dim=1)  # [B, D]
        
        # Normalize
        semantic_mean = F.normalize(semantic_mean, p=2, dim=1)
        timbre_mean = F.normalize(timbre_mean, p=2, dim=1)
        
        # Compute cosine similarity
        sim = torch.sum(semantic_mean * timbre_mean, dim=1).mean()
        
        # We want to minimize similarity (orthogonality)
        return sim
    
    def get_parameters(self):
        """Get all trainable parameters"""
        return (
            list(self.speaker_classifier.parameters()) +
            list(self.semantic_predictor.parameters()) +
            list(self.speaker_projector.parameters())
        )

    def __call__(self, global_speaker_emb, semantic_repr, timbre_tokens, speaker_ids):
        """Compute combined loss"""
        speaker_loss = self.get_speaker_loss(global_speaker_emb, speaker_ids)
        contrastive_loss = self.get_contrastive_loss(global_speaker_emb, semantic_repr)
        semantic_loss = self.get_semantic_consistency_loss(semantic_repr, timbre_tokens)
        
        # Weighted sum of losses
        total_loss = (
            self.speaker_weight * speaker_loss + 
            self.contrastive_weight * contrastive_loss + 
            self.semantic_weight * semantic_loss
        )
        
        return {
            'total_loss': total_loss,
            'speaker_loss': speaker_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'semantic_loss': semantic_loss.item()
        }