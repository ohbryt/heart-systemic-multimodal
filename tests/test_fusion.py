import pytest
import torch

from src.models.fusion import ModalityFusion

MODALITIES = ["bulk_rna", "scrna", "spatial", "proteomics", "metabolomics", "epigenomics"]


def test_fusion_all_modalities_present():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=128)
    embeddings = {mod: torch.randn(50, 128) for mod in MODALITIES}
    out = fusion(embeddings)
    assert out.shape == (50, 128)


def test_fusion_missing_modalities():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=128)
    embeddings = {
        "bulk_rna": torch.randn(50, 128),
        "proteomics": torch.randn(50, 128),
    }
    out = fusion(embeddings)
    assert out.shape == (50, 128)


def test_fusion_single_modality():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=128)
    embeddings = {"bulk_rna": torch.randn(50, 128)}
    out = fusion(embeddings)
    assert out.shape == (50, 128)


def test_fusion_different_fused_dim():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=64)
    embeddings = {mod: torch.randn(30, 128) for mod in MODALITIES}
    out = fusion(embeddings)
    assert out.shape == (30, 64)
