from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class PPIPathwayGNN(nn.Module):
    """GAT-based GNN for PPI + pathway graph with residual connections."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.2,
        n_edge_types: int = 2,
    ):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(n_edge_types, in_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(layers):
            in_channels = in_dim if i == 0 else hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=in_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: Data) -> torch.Tensor:
        edge_index = graph.edge_index
        edge_type = graph.edge_type if hasattr(graph, "edge_type") else torch.zeros(
            edge_index.shape[1], dtype=torch.long, device=x.device
        )

        if edge_index.shape[1] > 0:
            edge_attr = self.edge_type_embedding(edge_type)
        else:
            edge_attr = None

        for conv, norm in zip(self.convs, self.norms):
            residual = x if x.shape[1] == conv.out_channels * conv.heads else None
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            if residual is not None:
                x = x + residual
            x = torch.relu(x)
            x = self.dropout(x)

        return self.output_proj(x)
