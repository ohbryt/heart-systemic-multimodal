from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """2-layer MLP encoder for a single modality."""

    def __init__(self, in_features: int, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityEncoderBank(nn.Module):
    """Collection of per-modality encoders."""

    def __init__(self, modality_features: Dict[str, int], out_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.encoders = nn.ModuleDict({
            mod: ModalityEncoder(n_feat, hidden_dim, out_dim)
            for mod, n_feat in modality_features.items()
        })

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        if modality not in self.encoders:
            raise KeyError(f"Unknown modality: {modality}")
        return self.encoders[modality](x)
