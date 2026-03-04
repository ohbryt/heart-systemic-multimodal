import pytest
import torch
import pandas as pd

from src.training.evaluator import IrisEvaluator


def test_evaluate_returns_metrics():
    scores = torch.randn(100)
    labels = torch.zeros(100)
    labels[:20] = 1.0

    evaluator = IrisEvaluator()
    metrics = evaluator.evaluate(scores, labels)

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "ndcg_20" in metrics
    assert "ndcg_50" in metrics
    assert "hit_20" in metrics


def test_perfect_ranking():
    scores = torch.cat([torch.ones(20) * 10, torch.ones(80) * -10])
    labels = torch.cat([torch.ones(20), torch.zeros(80)])

    evaluator = IrisEvaluator()
    metrics = evaluator.evaluate(scores, labels)

    assert metrics["auroc"] > 0.99
    assert metrics["auprc"] > 0.99
    assert metrics["hit_20"] == 1.0


def test_export_results():
    scores = torch.randn(50)
    labels = torch.zeros(50)
    labels[:10] = 1.0
    gene_ids = list(range(1, 51))

    evaluator = IrisEvaluator()
    df = evaluator.export_ranked_list(scores, gene_ids, labels)
    assert isinstance(df, pd.DataFrame)
    assert "gene_id" in df.columns
    assert "score" in df.columns
    assert "rank" in df.columns
    assert df["rank"].iloc[0] == 1
    assert len(df) == 50
