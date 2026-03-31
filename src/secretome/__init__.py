"""
Secretome module for cardiac secretome inference.

Identifies candidate secreted ligands from disease-associated
differential expression and intersects with EV marker annotations.
"""

from .secretome_inference import CardiacSecretomeInference, EV_MARKERS

__all__ = [
    "CardiacSecretomeInference",
    "EV_MARKERS",
]
