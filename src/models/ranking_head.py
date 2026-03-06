from __future__ import annotations

import torch
import torch.nn as nn


class KeyDriverRankingHead(nn.Module):
    """MLP head that produces a scalar key-driver score per gene."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self._loss_fn: nn.BCEWithLogitsLoss | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def compute_loss(
        self, scores: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._loss_fn is None or pos_weight is not None:
            self._loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight,
            )
        return self._loss_fn(scores, labels)
