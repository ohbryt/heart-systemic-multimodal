"""
Curated gene set library for the Heart-Systemic Multi-modal Pipeline.

Provides named gene sets used for module scoring (scanpy.tl.score_genes),
EV cargo annotation, and cross-tissue comparison.

Gene sets are intentionally kept as plain module-level constants so that they
can be imported with zero overhead and are easily auditable.  Use the public
helper functions (:func:`get_gene_set`, :func:`get_all_sets`,
:func:`list_gene_sets`) for programmatic access.
"""

from __future__ import annotations

import logging
from typing import FrozenSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated gene sets
# ---------------------------------------------------------------------------

#: Fibrosis / ECM remodeling — TGF-b pathway + structural ECM genes.
FIBROSIS_ECM: FrozenSet[str] = frozenset({
    "COL1A1", "COL1A2", "COL3A1", "COL5A1",
    "FN1", "POSTN",
    "TGFB1", "TGFBR1", "TGFBR2",
    "CTGF",               # CCN2 alias
    "LOX", "LOXL2",
    "SPARC", "LUM", "DCN",
    "MMP2", "MMP9", "TIMP1",
})

#: Inflammatory signaling — NF-kB / cytokine axis.
INFLAMMATION: FrozenSet[str] = frozenset({
    "IL6", "TNF",
    "NFKB1", "RELA",
    "CCL2", "CXCL8", "CXCL2",
    "TLR2", "TLR4",
    "IL1B",
})

#: Cardiac hypertrophy / fetal gene program.
HYPERTROPHY: FrozenSet[str] = frozenset({
    "NPPA", "NPPB",
    "MYH7", "ACTA1",
    "GATA4", "MEF2C",
    "CAMK2D",
})

#: Mitochondrial biogenesis and oxidative phosphorylation stress.
MITO_STRESS: FrozenSet[str] = frozenset({
    "PPARGC1A",   # PGC-1α
    "TFAM", "NRF1",
    "NDUFS1",     # Complex I subunit
    "COX4I1",     # Complex IV subunit
    "ATP5F1A",    # ATP synthase subunit α
})

#: Cellular senescence and SASP (senescence-associated secretory phenotype).
SENESCENCE_SASP: FrozenSet[str] = frozenset({
    "CDKN2A",     # p16-INK4a
    "CDKN1A",     # p21
    "IL6", "IL8",
    "MMP3", "MMP9",
    "SERPINE1",   # PAI-1
})

#: Secretory / cardiokine program — growth factors and angiogenic signals.
SECRETORY: FrozenSet[str] = frozenset({
    "VEGFA", "FGF2",
    "ANGPT2", "PDGFB",
    "IGF1", "HGF",
})

#: Extracellular compartment — merged ECM structural proteins,
#: secreted cytokines, and growth factors.
EXTRACELLULAR: FrozenSet[str] = frozenset(
    FIBROSIS_ECM
    | SECRETORY
    | {
        # Additional cytokines / growth factors not covered above
        "IL6", "IL1B", "TNF",
        "CXCL12", "CCL2",
        "TGFB1", "BMP2", "BMP4",
        "WNT5A", "FGF21",
        "ANGPT1", "VEGFB",
        "FSTL1", "GDF15",
        "APLN", "METRNL",
    }
)

#: Extracellular vesicle (EV) biogenesis and surface markers.
EV_RELATED: FrozenSet[str] = frozenset({
    "CD63", "CD81", "CD9",         # Tetraspanin surface markers
    "TSG101", "ALIX",              # ESCRT pathway
    "SDCBP",                       # Syntenin-1 — exosome biogenesis
    "HSP90AA1",                    # Chaperone enriched in EVs
})

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Internal registry mapping canonical name → frozenset.
# Names are uppercase to match the convention used in YAML configs and logs.
_REGISTRY: dict[str, FrozenSet[str]] = {
    "FIBROSIS_ECM": FIBROSIS_ECM,
    "INFLAMMATION": INFLAMMATION,
    "HYPERTROPHY": HYPERTROPHY,
    "MITO_STRESS": MITO_STRESS,
    "SENESCENCE_SASP": SENESCENCE_SASP,
    "SECRETORY": SECRETORY,
    "EXTRACELLULAR": EXTRACELLULAR,
    "EV_RELATED": EV_RELATED,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_gene_set(name: str) -> list[str]:
    """Return a sorted list of gene symbols for the named gene set.

    Args:
        name: Case-insensitive name of the gene set
            (e.g. ``"FIBROSIS_ECM"``, ``"inflammation"``).

    Returns:
        Sorted list of HGNC gene symbols.

    Raises:
        KeyError: If *name* is not in the registry.

    Example::

        genes = get_gene_set("FIBROSIS_ECM")
        # → ["COL1A1", "COL1A2", ..., "TGFB1", "TIMP1"]
    """
    key = name.upper()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Gene set '{name}' not found. Available sets: {available}"
        )
    gene_list = sorted(_REGISTRY[key])
    logger.debug("get_gene_set('%s') → %d genes", key, len(gene_list))
    return gene_list


def list_gene_sets() -> list[str]:
    """Return a sorted list of all registered gene set names.

    Returns:
        Sorted list of gene set name strings.

    Example::

        names = list_gene_sets()
        # → ["EV_RELATED", "EXTRACELLULAR", "FIBROSIS_ECM", ...]
    """
    return sorted(_REGISTRY.keys())


def get_all_sets() -> dict[str, list[str]]:
    """Return all gene sets as a dict mapping name → sorted gene list.

    Suitable for passing directly to ``scanpy.tl.score_genes`` in a loop,
    or for serialising to JSON/YAML.

    Returns:
        Dict of ``{set_name: [gene1, gene2, ...]}`` with genes sorted
        alphabetically within each set.

    Example::

        all_sets = get_all_sets()
        for name, genes in all_sets.items():
            adata.obs[f"score_{name}"] = sc.tl.score_genes(adata, genes)
    """
    return {name: sorted(genes) for name, genes in _REGISTRY.items()}


def gene_set_sizes() -> dict[str, int]:
    """Return a dict mapping gene set name → number of genes.

    Useful for quick validation and logging.

    Returns:
        Dict of ``{set_name: gene_count}``.
    """
    return {name: len(genes) for name, genes in _REGISTRY.items()}


def genes_in_set(gene: str, name: str) -> bool:
    """Check whether a single gene symbol belongs to the named gene set.

    Args:
        gene: HGNC gene symbol (e.g. ``"COL1A1"``).
        name: Gene set name (case-insensitive).

    Returns:
        ``True`` if *gene* is a member of the set, ``False`` otherwise.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    return gene.upper() in {g.upper() for g in _REGISTRY[name.upper()]}
