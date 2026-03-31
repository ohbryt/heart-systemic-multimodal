"""
Scoring module for cardiac phenotype analysis.

Provides gene set scoring, differential expression, and related utilities
for single-cell RNA-seq data from cardiac tissues.
"""

from .phenotype_scorer import CardiacPhenotypeScorer, CARDIAC_GENE_SETS
from .differential import DifferentialExpression

__all__ = [
    "CardiacPhenotypeScorer",
    "CARDIAC_GENE_SETS",
    "DifferentialExpression",
]
