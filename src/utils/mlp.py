import torch.nn as nn

class SpeakerClassifier(nn.Module):
    """Simple speaker classifier using the global speaker embedding"""
    def __init__(self, emb_dim, num_speakers):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_speakers)
        )
        
    def forward(self, x):
        return self.classifier(x)


class SemanticPredictor(nn.Module):
    """Semantic predictor for contrastive learning"""
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.predictor(x)