from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.encoders import ModalityEncoderBank
from src.models.fusion import ModalityFusion
from src.models.gnn import PPIPathwayGNN
from src.models.ranking_head import KeyDriverRankingHead


class IrisModel(nn.Module):
    """Full IRIS pipeline: encoders -> fusion -> GNN -> ranking."""

    def __init__(
        self,
        modality_features: Dict[str, int],
        embed_dim: int = 128,
        fused_dim: int = 128,
        gnn_hidden: int = 128,
        gnn_out: int = 128,
        gnn_heads: int = 4,
        gnn_layers: int = 3,
        ranking_hidden: int = 64,
    ):
        super().__init__()
        self.modality_names = list(modality_features.keys())

        self.encoder_bank = ModalityEncoderBank(
            modality_features=modality_features,
            out_dim=embed_dim,
        )
        self.fusion = ModalityFusion(
            modality_names=self.modality_names,
            embed_dim=embed_dim,
            fused_dim=fused_dim,
        )
        self.gnn = PPIPathwayGNN(
            in_dim=fused_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            heads=gnn_heads,
            layers=gnn_layers,
        )
        self.ranking_head = KeyDriverRankingHead(
            in_dim=gnn_out,
            hidden_dim=ranking_hidden,
        )

    def forward(
        self, modality_data: Dict[str, torch.Tensor], graph: Data
    ) -> torch.Tensor:
        embeddings = {}
        for mod, features in modality_data.items():
            if mod in self.modality_names:
                embeddings[mod] = self.encoder_bank(mod, features)

        fused = self.fusion(embeddings)
        refined = self.gnn(fused, graph)
        return self.ranking_head(refined)
