# app/model.py
import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    """
    x: (B, T, F)
    Conv1d opera sobre T con canales=F.
    Modelo pequeño => entrena rápido en CPU.
    """
    def __init__(self, feature_dim: int, num_classes: int = 3, hidden: int = 48, dropout: float = 0.15):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),  # -> (B, hidden, 1)
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        h = self.net(x).squeeze(-1)   # (B, hidden)
        return self.fc(h)             # (B, C)