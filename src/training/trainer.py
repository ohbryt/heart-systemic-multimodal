from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from src.models.iris_model import IrisModel

logger = logging.getLogger(__name__)


@dataclass
class IrisTrainer:
    """Training loop for the IRIS model."""

    model: IrisModel
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    val_fraction: float = 0.2
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

        # Compute class weight for imbalanced labels
        n_pos = labels[train_idx].sum().item()
        n_neg = len(train_idx) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        logger.info("Class balance: %d pos, %d neg, pos_weight=%.2f", int(n_pos), int(n_neg), pos_weight.item())

        optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            scores = self.model(modality_data, graph)

            loss = self.model.ranking_head.compute_loss(
                scores[train_idx], labels[train_idx], pos_weight=pos_weight
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            history["train_loss"].append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_scores = self.model(modality_data, graph)[val_idx]
                val_loss = self.model.ranking_head.compute_loss(
                    val_scores, labels[val_idx], pos_weight=pos_weight
                )
                history["val_loss"].append(val_loss.item())

            scheduler.step(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            if (epoch + 1) % 10 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d: train_loss=%.4f val_loss=%.4f lr=%.2e",
                    epoch + 1, loss.item(), val_loss.item(), lr_now,
                )

        return history
