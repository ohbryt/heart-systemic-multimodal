"""Literature-based sarcopenia gene labels for IRIS supervision."""
from __future__ import annotations

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)

# Curated sarcopenia / muscle-atrophy associated genes from literature.
# Sources: OMIM, GeneCards, recent GWAS/meta-analyses, review papers.
SARCOPENIA_GENES: set[str] = {
    # Myostatin / TGF-beta signaling
    "MSTN", "GDF8",  # myostatin (GDF8 is alias)
    "TGFB1", "TGFBR1", "TGFBR2",
    "ACVR2B", "ACVR2A", "ACVR1B",
    "SMAD2", "SMAD3", "SMAD4", "SMAD7",
    "GDF11", "BMP7", "INHBA",
    # IGF-1 / PI3K-Akt / mTOR signaling
    "IGF1", "IGF1R", "IGF2", "IGFBP3", "IGFBP5",
    "IRS1", "IRS2",
    "PIK3CA", "PIK3CB", "PIK3R1",
    "AKT1", "AKT2",
    "MTOR", "RPTOR", "RICTOR", "RPS6KB1", "EIF4EBP1",
    # Ubiquitin-proteasome (muscle atrophy)
    "TRIM63", "MURF1",  # MuRF1 (TRIM63 is official)
    "FBXO32", "ATROGIN1", "MAFBX",  # Atrogin-1/MAFbx (FBXO32 is official)
    "FBXO30", "MUSA1",
    "UBR5", "ASB2",
    # Autophagy-lysosome
    "ATG5", "ATG7", "ATG12",
    "BECN1", "BNIP3", "BNIP3L",
    "CTSL", "LAMP2",
    # FOXO transcription factors
    "FOXO1", "FOXO3", "FOXO4",
    # AMPK subunits
    "PRKAA1", "PRKAA2", "PRKAB1", "PRKAG1",
    # PGC-1alpha / mitochondrial
    "PPARGC1A", "PPARGC1B",  # PGC-1alpha, PGC-1beta
    "TFAM", "NRF1",
    "SIRT1", "SIRT3",
    # Satellite cell / myogenesis
    "PAX7", "PAX3",
    "MYOD1", "MYF5", "MYF6", "MYOG",
    # Myosin heavy chains
    "MYH1", "MYH2", "MYH4", "MYH7",
    # Inflammatory cytokines
    "IL6", "IL6R", "TNF", "TNFRSF1A",
    "IL1B", "IL15", "IL10",
    # HDAC / epigenetic regulators
    "HDAC4", "HDAC5", "HDAC9",
    # Calcium / structural
    "RYR1", "CASQ1", "DMD", "DES",
    "TTN", "ACTA1", "LMNA",
    # Apoptosis / NF-kB
    "NFKB1", "RELA", "CASP3",
    # Growth/differentiation
    "FGF2", "HGF", "LEP", "LEPR",
    "GH1", "GHR",
    # Vitamin D receptor
    "VDR",
    # ACE (exercise response)
    "ACE", "ACTN3",
}

# Build a case-insensitive lookup with aliases
_GENE_LOOKUP: set[str] = {g.upper() for g in SARCOPENIA_GENES}


def build_labels(
    gene_names: List[str] | None,
    gene_ids: List[int],
    seed: int = 42,
) -> torch.Tensor:
    """Build supervision labels from literature-based sarcopenia gene set.

    Args:
        gene_names: Gene symbols (e.g., ["MSTN", "IGF1", ...]). If None, falls back to random.
        gene_ids: Integer gene IDs (used for fallback and sizing).
        seed: Random seed for fallback labels.

    Returns:
        Float tensor of shape (n_genes,) with 1.0 for positive, 0.0 for negative.
    """
    n_genes = len(gene_ids)

    if gene_names is None or len(gene_names) == 0:
        logger.warning("No gene names provided, falling back to random labels")
        return _random_labels(n_genes, seed)

    labels = torch.zeros(n_genes)
    matched = []

    for i, name in enumerate(gene_names):
        if name.upper() in _GENE_LOOKUP:
            labels[i] = 1.0
            matched.append(name)

    n_matched = int(labels.sum().item())
    logger.info(
        "Label builder: %d/%d genes matched sarcopenia set (%.1f%%)",
        n_matched, n_genes, 100.0 * n_matched / n_genes,
    )
    if matched:
        logger.info("Matched genes: %s", ", ".join(sorted(matched)[:20]))
        if len(matched) > 20:
            logger.info("  ... and %d more", len(matched) - 20)

    if n_matched < 5:
        logger.warning(
            "Only %d matches found (<5). Falling back to random labels. "
            "Check that gene_names contain standard HGNC symbols.",
            n_matched,
        )
        return _random_labels(n_genes, seed)

    return labels


def _random_labels(n_genes: int, seed: int) -> torch.Tensor:
    """Fallback: random 20% positive labels."""
    # Use seed + 7 to avoid collision with trainer's train/val split (same seed)
    rng = torch.Generator().manual_seed(seed + 7)
    labels = torch.zeros(n_genes)
    n_pos = max(1, n_genes // 5)
    pos_idx = torch.randperm(n_genes, generator=rng)[:n_pos]
    labels[pos_idx] = 1.0
    return labels
