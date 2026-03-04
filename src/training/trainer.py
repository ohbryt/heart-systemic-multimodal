from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data

from src.models.iris_model import IrisModel

logger = logging.getLogger(__name__)


@dataclass
class IrisTrainer:
    """Training loop for the IRIS model."""

    model: IrisModel
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    val_fraction: float = 0.2
    n_neg_per_pos: int = 3
    seed: int = 42

    def train(
        self,
        modality_data: Dict[str, torch.Tensor],
        graph: Data,
        labels: torch.Tensor,
    ) -> Dict[str, List[float]]:
        torch.manual_seed(self.seed)

        n = labels.shape[0]
        perm = torch.randperm(n)
        val_size = int(n * self.val_fraction)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            scores = self.model(modality_data, graph)

            train_scores = scores[train_idx]
            train_labels = labels[train_idx]
            loss = self._pairwise_loss(train_scores, train_labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            history["train_loss"].append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_scores = self.model(modality_data, graph)[val_idx]
                val_labels = labels[val_idx]
                val_loss = self._pairwise_loss(val_scores, val_labels)
                history["val_loss"].append(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.4f val_loss=%.4f",
                    epoch + 1, loss.item(), val_loss.item(),
                )

        return history

    def _pairwise_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_mask = labels > 0.5
        neg_mask = labels <= 0.5

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, requires_grad=True)

        n_neg = len(neg_scores)
        n_pairs = min(len(pos_scores) * self.n_neg_per_pos, n_neg)

        pos_expanded = pos_scores.repeat_interleave(self.n_neg_per_pos)[:n_pairs]
        neg_sampled = neg_scores[torch.randint(n_neg, (n_pairs,))]

        # Ensure same size
        size = min(len(pos_expanded), len(neg_sampled))
        return self.model.ranking_head.compute_loss(pos_expanded[:size], neg_sampled[:size])
