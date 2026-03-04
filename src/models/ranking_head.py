from __future__ import annotations

import torch
import torch.nn as nn


class KeyDriverRankingHead(nn.Module):
    """MLP head that produces a scalar key-driver score per gene."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 64, margin: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def compute_loss(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor
    ) -> torch.Tensor:
        target = torch.ones_like(pos_scores)
        return self.margin_loss(pos_scores, neg_scores, target)
