from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class IrisEvaluator:
    """Evaluate IRIS model predictions."""

    def evaluate(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        s = scores.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        metrics = {}
        metrics["auroc"] = float(roc_auc_score(y, s))
        metrics["auprc"] = float(average_precision_score(y, s))

        for k in [20, 50, 100]:
            metrics[f"ndcg_{k}"] = self._ndcg_at_k(s, y, k)

        for k in [20, 50]:
            metrics[f"hit_{k}"] = self._hit_at_k(s, y, k)

        return metrics

    def _ndcg_at_k(self, scores: np.ndarray, labels: np.ndarray, k: int) -> float:
        k = min(k, len(scores))
        order = np.argsort(-scores)[:k]
        dcg = np.sum(labels[order] / np.log2(np.arange(2, k + 2)))
        ideal_order = np.argsort(-labels)[:k]
        idcg = np.sum(labels[ideal_order] / np.log2(np.arange(2, k + 2)))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def _hit_at_k(self, scores: np.ndarray, labels: np.ndarray, k: int) -> float:
        k = min(k, len(scores))
        top_k_idx = np.argsort(-scores)[:k]
        n_pos_in_topk = labels[top_k_idx].sum()
        n_pos_total = labels.sum()
        return float(n_pos_in_topk / n_pos_total) if n_pos_total > 0 else 0.0

    def export_ranked_list(
        self,
        scores: torch.Tensor,
        gene_ids: List[int],
        labels: Optional[torch.Tensor] = None,
    ) -> pd.DataFrame:
        s = scores.detach().cpu().numpy()
        order = np.argsort(-s)

        df = pd.DataFrame({
            "gene_id": [gene_ids[i] for i in order],
            "score": np.round(s[order], 4),
            "rank": np.arange(1, len(order) + 1),
        })

        if labels is not None:
            y = labels.detach().cpu().numpy()
            df["evidence_tier"] = ["positive" if y[i] > 0.5 else "negative" for i in order]

        return df
