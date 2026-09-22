import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    """Encode normalized age and binary sex features into a dense embedding."""

    def __init__(self, input_dim=2, hidden_dim=32, output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return clinical embeddings with shape ``[batch_size, output_dim]``."""
        return self.network(x)