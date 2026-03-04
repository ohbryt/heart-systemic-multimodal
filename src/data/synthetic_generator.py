from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class ModalityData:
    """Standardized container for one modality's data."""
    features: torch.Tensor  # (n_genes, n_features)
    gene_ids: List[int]     # Entrez-style integer IDs
    modality: str

    @property
    def n_genes(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


# Feature dimensions per modality (realistic ranges)
MODALITY_FEATURES = {
    "bulk_rna": 500,
    "scrna": 300,
    "spatial": 200,
    "proteomics": 150,
    "metabolomics": 80,
    "epigenomics": 250,
}


@dataclass
class SyntheticDataGenerator:
    """Generate realistic synthetic multi-omics data for model development."""
    n_genes: int = 500
    seed: int = 42

    def generate(self) -> Dict[str, ModalityData]:
        rng = torch.Generator().manual_seed(self.seed)
        gene_ids = list(range(1, self.n_genes + 1))
        data = {}
        for modality, n_feat in MODALITY_FEATURES.items():
            features = self._generate_modality(modality, n_feat, rng)
            data[modality] = ModalityData(
                features=features,
                gene_ids=gene_ids,
                modality=modality,
            )
        return data

    def _generate_modality(
        self, modality: str, n_features: int, rng: torch.Generator
    ) -> torch.Tensor:
        shape = (self.n_genes, n_features)
        if modality in ("bulk_rna", "scrna", "spatial"):
            raw = torch.zeros(shape).normal_(mean=2.0, std=1.5, generator=rng)
            return raw.exp().clamp(min=0)
        elif modality == "epigenomics":
            raw = torch.zeros(shape).normal_(mean=0.0, std=1.0, generator=rng)
            return raw.sigmoid()
        elif modality == "proteomics":
            raw = torch.zeros(shape).normal_(mean=1.0, std=2.0, generator=rng)
            features = raw.exp().clamp(min=0)
            mask = torch.zeros(shape).bernoulli_(0.7, generator=rng)
            return features * mask
        elif modality == "metabolomics":
            raw = torch.zeros(shape).normal_(mean=1.5, std=1.0, generator=rng)
            return raw.exp().clamp(min=0)
        else:
            return torch.zeros(shape).uniform_(0, 1, generator=rng)
