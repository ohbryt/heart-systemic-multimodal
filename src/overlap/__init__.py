"""
Overlap module for cross-referencing cardiac secretome with plasma proteomics.

Intersects candidate secreted ligands with proteins detected in plasma EV
proteomics datasets (PXD021371, PXD059929, PXD060680) and tests statistical
enrichment.
"""

from .plasma_overlap import PlasmaOverlapAnalyzer, PLASMA_DATASET_IDS

__all__ = [
    "PlasmaOverlapAnalyzer",
    "PLASMA_DATASET_IDS",
]
