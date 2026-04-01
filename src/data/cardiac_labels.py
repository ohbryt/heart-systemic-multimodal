"""Literature-based cardiac/HF gene labels for IRIS supervision."""
from __future__ import annotations

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)

# Curated heart failure / cardiac remodeling genes from literature.
# Sources: OMIM, ClinVar, GWAS Catalog, AHA/ESC guidelines, recent reviews.
CARDIAC_GENES: set[str] = {
    # Contractile apparatus
    "MYH7", "MYH6", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3",
    "MYBPC3", "TNNC1", "MYH11",
    # Sarcomeric / structural
    "TTN", "LMNA", "DMD", "DES", "VCL", "FLNC", "BAG3", "CSRP3",
    "ACTN2", "LDB3", "TCAP", "MYPN",
    # Calcium handling
    "RYR2", "PLN", "CASQ2", "ATP2A2", "CALM1", "CALM2", "CALM3",
    "CACNA1C", "SLC8A1",
    # Ion channels / arrhythmia
    "SCN5A", "KCNQ1", "KCNH2", "HCN4", "KCNJ2", "KCNE1", "KCNE2",
    # Natriuretic peptides / biomarkers
    "NPPB", "NPPA", "MYH7", "TNNT2", "GDF15", "WFDC2",
    "PTX3", "IGFBP2", "CHI3L1",
    # RAAS pathway
    "ACE", "ACE2", "AGT", "AGTR1", "AGTR2", "REN",
    # Apelin system
    "APLN", "APLNR",
    # Adrenergic signaling
    "ADRB1", "ADRB2", "GRK2", "GRK5", "ARRB1", "ARRB2",
    # Fibrosis / ECM remodeling
    "COL1A1", "COL1A2", "COL3A1", "POSTN", "CTGF", "CCN2",
    "TNC", "FN1", "LOX", "LOXL2", "FAP",
    "MMP2", "MMP9", "MMP14", "TIMP1", "TIMP2",
    # TGF-beta / BMP
    "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
    "SMAD2", "SMAD3", "SMAD4", "SMAD7",
    "BMP2", "BMP4", "BMP7", "GDF11", "MSTN",
    # Wnt signaling (cardiac)
    "WNT9A", "WNT3A", "WNT5A", "CTNNB1", "GSK3B",
    "SFRP1", "SFRP2", "DKK1", "DKK3",
    # Inflammatory / NF-kB
    "IL6", "IL6R", "IL1B", "TNF", "TNFRSF1A",
    "NFKB1", "RELA", "NLRP3", "IL18", "IL33", "IL1RL1",
    # Oxidative stress
    "SOD2", "CAT", "GPX1", "HMOX1", "NOX2", "NOX4", "NOS3",
    # Mitochondrial
    "PPARGC1A", "TFAM", "SIRT1", "SIRT3",
    "MTOR", "RPTOR", "PRKAA1", "PRKAA2",
    # Apoptosis
    "BAX", "BCL2", "CASP3", "CASP9", "TP53",
    # Hypertrophy signaling
    "NFATC1", "NFATC2", "GATA4", "MEF2A", "MEF2C",
    "MAPK1", "MAPK3", "MAPK14", "CAMK2D",
    "JAK2", "STAT3",
    # Endothelial / angiogenesis
    "VEGFA", "KDR", "FLT1", "PECAM1", "CDH5", "VWF",
    "NOTCH1", "DLL4", "ENG", "ANGPT1", "ANGPT2", "TEK",
    # Cardiac transcription factors
    "NKX2-5", "TBX5", "TBX20", "HAND1", "HAND2", "IRX4",
    # Metabolism
    "PPARA", "PPARG", "PPARD", "RXRA",
    "CPT1B", "CPT2", "ACADVL", "CD36", "SLC2A4",
    # Autophagy (cardiac)
    "ATG5", "ATG7", "BECN1", "BNIP3", "LAMP2",
    # Cardiac-specific from pressure overload data
    "CRIM1", "SEMA4C", "LAYN", "ITGA6", "SPP1", "DCN",
    "LTBP2", "OLR1", "TLR4", "FPR1",
    "ENPP2", "AGRN", "ACTA2", "VIM",
}

_CARDIAC_LOOKUP: set[str] = {g.upper() for g in CARDIAC_GENES}


def build_cardiac_labels(
    gene_names: List[str] | None,
    gene_ids: List[int],
    seed: int = 42,
) -> torch.Tensor:
    """Build supervision labels from cardiac/HF gene set.

    Returns:
        Float tensor of shape (n_genes,) with 1.0 for cardiac genes, 0.0 otherwise.
    """
    n_genes = len(gene_ids)

    if gene_names is None or len(gene_names) == 0:
        logger.warning("No gene names provided, falling back to random labels")
        return _random_labels(n_genes, seed)

    labels = torch.zeros(n_genes)
    matched = []

    for i, name in enumerate(gene_names):
        if name.upper() in _CARDIAC_LOOKUP:
            labels[i] = 1.0
            matched.append(name)

    n_matched = int(labels.sum().item())
    logger.info(
        "Cardiac label builder: %d/%d genes matched cardiac set (%.1f%%)",
        n_matched, n_genes, 100.0 * n_matched / n_genes,
    )
    if matched:
        logger.info("Matched cardiac genes: %s", ", ".join(sorted(matched)[:20]))
        if len(matched) > 20:
            logger.info("  ... and %d more", len(matched) - 20)

    if n_matched < 5:
        logger.warning(
            "Only %d cardiac matches (<5). Falling back to random labels.", n_matched,
        )
        return _random_labels(n_genes, seed)

    return labels


def _random_labels(n_genes: int, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed + 7)
    labels = torch.zeros(n_genes)
    n_pos = max(1, n_genes // 5)
    pos_idx = torch.randperm(n_genes, generator=rng)[:n_pos]
    labels[pos_idx] = 1.0
    return labels
