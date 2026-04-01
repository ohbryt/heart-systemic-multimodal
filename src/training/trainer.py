from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import KFold
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

    def __post_init__(self):
        self._model_config = self._extract_model_config()

    def _extract_model_config(self) -> Dict:
        """Extract model architecture config for recreation."""
        name = list(self.model.encoder_bank.encoders.keys())[0]
        mod_features = {
            n: e.net[0].in_features for n, e in self.model.encoder_bank.encoders.items()
        }
        return {
            "modality_features": mod_features,
            "embed_dim": self.model.encoder_bank.encoders[name].net[4].out_features,
            "fused_dim": self.model.fusion.projection.out_features,
            "gnn_hidden": self.model.gnn.convs[0].out_channels
            * self.model.gnn.convs[0].heads,
            "gnn_out": self.model.gnn.output_proj.out_features
            if hasattr(self.model.gnn.output_proj, "out_features")
            else self.model.gnn.convs[0].out_channels * self.model.gnn.convs[0].heads,
            "gnn_heads": self.model.gnn.convs[0].heads,
            "gnn_layers": len(self.model.gnn.convs),
            "ranking_hidden": self.model.ranking_head.net[0].out_features,
        }

    def _create_fresh_model(self) -> IrisModel:
        """Create a fresh model with same architecture."""
        return IrisModel(**self._model_config)

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
        logger.info(
            "Class balance: %d pos, %d neg, pos_weight=%.2f",
            int(n_pos),
            int(n_neg),
            pos_weight.item(),
        )

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
                    epoch + 1,
                    loss.item(),
                    val_loss.item(),
                    lr_now,
                )

        return history

    def cross_validate(
        self,
        modality_data: Dict[str, torch.Tensor],
        graph: Data,
        labels: torch.Tensor,
        k: int = 5,
    ) -> Dict:
        """K-fold cross-validation.

        Returns:
            {
                "fold_metrics": [{auroc, auprc, ndcg_20, ...}, ...],
                "mean_metrics": {auroc_mean, auroc_std, ...},
            }
        """
        from src.training.evaluator import IrisEvaluator

        torch.manual_seed(self.seed)
        n = labels.shape[0]

        kfold = KFold(n_splits=k, shuffle=True, random_state=self.seed)
        evaluator = IrisEvaluator()

        fold_metrics = []
        all_predictions = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(torch.arange(n))):
            logger.info(f"=== Fold {fold_idx + 1}/{k} ===")

            # Clone model for each fold
            model = self._create_fresh_model()

            # Compute pos_weight
            n_pos = labels[train_idx].sum().item()
            n_neg = len(train_idx) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
            logger.info(
                f"  Fold {fold_idx + 1}: {n_pos} pos, {n_neg} neg, pos_weight={pos_weight.item():.2f}"
            )

            optimizer = AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.epochs):
                model.train()
                optimizer.zero_grad()
                scores = model(modality_data, graph)

                loss = model.ranking_head.compute_loss(
                    scores[train_idx], labels[train_idx], pos_weight=pos_weight
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_scores = model(modality_data, graph)[val_idx]
                    val_loss = model.ranking_head.compute_loss(
                        val_scores, labels[val_idx], pos_weight=pos_weight
                    )

                scheduler.step(val_loss.item())

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

            # Evaluate on validation fold
            model.eval()
            with torch.no_grad():
                val_scores = model(modality_data, graph)[val_idx]
                val_labels = labels[val_idx]

                # Skip fold if no positive labels
                if val_labels.sum() == 0:
                    logger.warning(
                        f"  Fold {fold_idx + 1}: No positive labels, skipping"
                    )
                    continue

                fold_result = evaluator.evaluate(val_scores, val_labels)
                fold_metrics.append(fold_result)
                all_predictions.append((val_scores, val_labels, fold_idx))

                logger.info(
                    "  Fold %d: AUROC=%.3f, AUPRC=%.3f, nDCG@20=%.3f",
                    fold_idx + 1,
                    fold_result["auroc"],
                    fold_result["auprc"],
                    fold_result["ndcg_20"],
                )

        # Compute mean ± std
        mean_metrics = {}
        for key in fold_metrics[0]:
            values = [fm[key] for fm in fold_metrics]
            mean_metrics[f"{key}_mean"] = np.mean(values)
            mean_metrics[f"{key}_std"] = np.std(values)

        logger.info("=" * 50)
        logger.info("CV Results (%d-fold):", k)
        logger.info(
            "  AUROC:  %.3f ± %.3f",
            mean_metrics["auroc_mean"],
            mean_metrics["auroc_std"],
        )
        logger.info(
            "  AUPRC:  %.3f ± %.3f",
            mean_metrics["auprc_mean"],
            mean_metrics["auprc_std"],
        )
        logger.info(
            "  nDCG@20: %.3f ± %.3f",
            mean_metrics["ndcg_20_mean"],
            mean_metrics["ndcg_20_std"],
        )
        logger.info(
            "  nDCG@50: %.3f ± %.3f",
            mean_metrics["ndcg_50_mean"],
            mean_metrics["ndcg_50_std"],
        )

        return {
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
            "all_predictions": all_predictions,
        }
