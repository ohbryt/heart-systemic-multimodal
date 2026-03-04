import pytest
import torch
from torch_geometric.data import Data

from src.models.iris_model import IrisModel
from src.data.synthetic_generator import SyntheticDataGenerator, MODALITY_FEATURES


def _make_graph(n_nodes):
    edge_index = torch.randint(0, n_nodes, (2, n_nodes * 5))
    edge_type = torch.randint(0, 2, (n_nodes * 5,))
    return Data(num_nodes=n_nodes, edge_index=edge_index, edge_type=edge_type)


def test_iris_forward_all_modalities():
    n_genes = 50
    model = IrisModel(modality_features=MODALITY_FEATURES)
    gen = SyntheticDataGenerator(n_genes=n_genes, seed=42)
    data = gen.generate()
    graph = _make_graph(n_genes)

    modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    scores = model(modality_tensors, graph)
    assert scores.shape == (n_genes,)


def test_iris_forward_missing_modalities():
    n_genes = 50
    model = IrisModel(modality_features=MODALITY_FEATURES)
    graph = _make_graph(n_genes)

    modality_tensors = {
        "bulk_rna": torch.randn(n_genes, MODALITY_FEATURES["bulk_rna"]),
        "proteomics": torch.randn(n_genes, MODALITY_FEATURES["proteomics"]),
    }
    scores = model(modality_tensors, graph)
    assert scores.shape == (n_genes,)


def test_iris_training_step():
    n_genes = 50
    model = IrisModel(modality_features=MODALITY_FEATURES)
    gen = SyntheticDataGenerator(n_genes=n_genes, seed=42)
    data = gen.generate()
    graph = _make_graph(n_genes)

    modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    scores = model(modality_tensors, graph)

    pos_idx = torch.arange(0, 25)
    neg_idx = torch.arange(25, 50)
    loss = model.ranking_head.compute_loss(scores[pos_idx], scores[neg_idx])

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
