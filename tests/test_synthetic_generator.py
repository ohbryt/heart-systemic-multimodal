import pytest
import torch

from src.data.synthetic_generator import SyntheticDataGenerator, ModalityData, MODALITY_FEATURES

MODALITIES = list(MODALITY_FEATURES.keys())


def test_generates_all_modalities():
    gen = SyntheticDataGenerator(n_genes=200, seed=42)
    data = gen.generate()
    assert isinstance(data, dict)
    for mod in MODALITIES:
        assert mod in data, f"Missing modality: {mod}"
        assert isinstance(data[mod], ModalityData)


def test_modality_data_shapes():
    gen = SyntheticDataGenerator(n_genes=100, seed=42)
    data = gen.generate()
    for mod in MODALITIES:
        md = data[mod]
        assert md.features.shape[0] == 100, f"{mod} gene count mismatch"
        assert md.features.shape[1] == MODALITY_FEATURES[mod], f"{mod} feature dim mismatch"
        assert len(md.gene_ids) == 100


def test_shared_gene_ids():
    gen = SyntheticDataGenerator(n_genes=100, seed=42)
    data = gen.generate()
    gene_ids = data["bulk_rna"].gene_ids
    for mod in MODALITIES:
        assert data[mod].gene_ids == gene_ids, f"{mod} gene IDs don't match"


def test_feature_distributions_are_nonnegative():
    gen = SyntheticDataGenerator(n_genes=100, seed=42)
    data = gen.generate()
    for mod in MODALITIES:
        assert (data[mod].features >= 0).all(), f"{mod} has negative values"


def test_deterministic_with_seed():
    g1 = SyntheticDataGenerator(n_genes=50, seed=99).generate()
    g2 = SyntheticDataGenerator(n_genes=50, seed=99).generate()
    for mod in MODALITIES:
        assert torch.equal(g1[mod].features, g2[mod].features)
