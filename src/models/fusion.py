from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ModalityFusion(nn.Module):
    """Fuse per-modality embeddings with missing-modality handling."""

    def __init__(self, modality_names: List[str], embed_dim: int = 128, fused_dim: int = 128):
        super().__init__()
        self.modality_names = modality_names
        self.embed_dim = embed_dim
        self.n_modalities = len(modality_names)

        self.projection = nn.Linear(self.n_modalities * embed_dim, fused_dim)
        self.presence_embedding = nn.Embedding(self.n_modalities, fused_dim)
        self.layer_norm = nn.LayerNorm(fused_dim)

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        sample = next(iter(embeddings.values()))
        n_genes = sample.shape[0]
        device = sample.device

        parts = []
        presence_indices = []
        for i, mod in enumerate(self.modality_names):
            if mod in embeddings:
                parts.append(embeddings[mod])
                presence_indices.append(i)
            else:
                parts.append(torch.zeros(n_genes, self.embed_dim, device=device))

        concat = torch.cat(parts, dim=1)
        projected = self.projection(concat)

        if presence_indices:
            idx = torch.tensor(presence_indices, device=device)
            presence_bias = self.presence_embedding(idx).mean(dim=0)
            projected = projected + presence_bias

        return self.layer_norm(projected)
