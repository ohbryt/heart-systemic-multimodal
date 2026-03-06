import pytest
import torch

from src.models.ranking_head import KeyDriverRankingHead


def test_ranking_head_output_shape():
    head = KeyDriverRankingHead(in_dim=128, hidden_dim=64)
    x = torch.randn(100, 128)
    scores = head(x)
    assert scores.shape == (100,)


def test_ranking_head_differentiable():
    head = KeyDriverRankingHead(in_dim=128, hidden_dim=64)
    x = torch.randn(50, 128, requires_grad=True)
    scores = head(x)
    loss = scores.sum()
    loss.backward()
    assert x.grad is not None


def test_ranking_loss():
    head = KeyDriverRankingHead(in_dim=128, hidden_dim=64)
    x = torch.randn(20, 128)
    scores = head(x)
    labels = torch.cat([torch.ones(10), torch.zeros(10)])
    loss = head.compute_loss(scores, labels)
    assert loss.item() >= 0
    assert loss.requires_grad
