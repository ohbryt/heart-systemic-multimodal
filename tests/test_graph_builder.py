import pytest
import torch
from torch_geometric.data import Data

from src.data.graph_builder import GraphBuilder


def test_build_synthetic_graph():
    gene_ids = list(range(1, 101))
    builder = GraphBuilder(use_api=False, seed=42)
    graph = builder.build(gene_ids)
    assert isinstance(graph, Data)
    assert graph.num_nodes == 100
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0
    assert hasattr(graph, "edge_type")
    assert graph.edge_type.shape[0] == graph.edge_index.shape[1]


def test_graph_edge_types():
    gene_ids = list(range(1, 51))
    builder = GraphBuilder(use_api=False, seed=42)
    graph = builder.build(gene_ids)
    assert set(graph.edge_type.unique().tolist()).issubset({0, 1})


def test_graph_is_undirected():
    gene_ids = list(range(1, 51))
    builder = GraphBuilder(use_api=False, seed=42)
    graph = builder.build(gene_ids)
    src, dst = graph.edge_index
    edges_forward = set(zip(src.tolist(), dst.tolist()))
    edges_backward = set(zip(dst.tolist(), src.tolist()))
    assert edges_forward == edges_backward


def test_graph_caching(tmp_path):
    gene_ids = list(range(1, 51))
    cache_path = tmp_path / "graph.pt"
    builder = GraphBuilder(use_api=False, seed=42, cache_path=str(cache_path))
    g1 = builder.build(gene_ids)
    assert cache_path.exists()
    g2 = builder.build(gene_ids)
    assert torch.equal(g1.edge_index, g2.edge_index)


def test_deterministic():
    gene_ids = list(range(1, 51))
    g1 = GraphBuilder(use_api=False, seed=42).build(gene_ids)
    g2 = GraphBuilder(use_api=False, seed=42).build(gene_ids)
    assert torch.equal(g1.edge_index, g2.edge_index)
