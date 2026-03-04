import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn import PPIPathwayGNN


def _make_test_graph(n_nodes: int = 50, n_edges: int = 200) -> Data:
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_type = torch.randint(0, 2, (n_edges,))
    return Data(num_nodes=n_nodes, edge_index=edge_index, edge_type=edge_type)


def test_gnn_output_shape():
    gnn = PPIPathwayGNN(in_dim=128, hidden_dim=128, out_dim=128, heads=4, layers=3)
    graph = _make_test_graph(50, 200)
    x = torch.randn(50, 128)
    out = gnn(x, graph)
    assert out.shape == (50, 128)


def test_gnn_single_layer():
    gnn = PPIPathwayGNN(in_dim=64, hidden_dim=64, out_dim=64, heads=2, layers=1)
    graph = _make_test_graph(30, 100)
    x = torch.randn(30, 64)
    out = gnn(x, graph)
    assert out.shape == (30, 64)


def test_gnn_no_edges():
    gnn = PPIPathwayGNN(in_dim=128, hidden_dim=128, out_dim=128, heads=4, layers=3)
    graph = Data(num_nodes=20, edge_index=torch.zeros(2, 0, dtype=torch.long),
                 edge_type=torch.zeros(0, dtype=torch.long))
    x = torch.randn(20, 128)
    out = gnn(x, graph)
    assert out.shape == (20, 128)


def test_gnn_gradient_flow():
    gnn = PPIPathwayGNN(in_dim=128, hidden_dim=128, out_dim=128, heads=4, layers=2)
    graph = _make_test_graph(30, 100)
    x = torch.randn(30, 128, requires_grad=True)
    out = gnn(x, graph)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)
