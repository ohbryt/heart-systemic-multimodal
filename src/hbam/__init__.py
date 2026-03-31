"""
HBAM — Heart-Based Aging and Multiorgan Index.

Aggregates per-sample feature vectors from phenotype scoring, secretome
analysis, plasma overlap detection, and cross-organ receptor mapping into
a single composite aging/disease index with ML-based prediction and SHAP
explainability.
"""

from .hbam_index import HBAMIndex
from .feature_builder import HBAMFeatureBuilder

__all__ = [
    "HBAMIndex",
    "HBAMFeatureBuilder",
]
