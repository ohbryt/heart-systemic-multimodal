"""
Preprocessing package for heart systemic multimodal project.

Provides preprocessing pipelines for:
- Heart single-cell RNA-seq (scanpy-based)
- Proteomics data from PRIDE
"""

from .heart_sc import HeartSCPreprocessor
from .proteomics import ProteomicsPreprocessor

__all__ = [
    "HeartSCPreprocessor",
    "ProteomicsPreprocessor",
]
