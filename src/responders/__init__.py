"""
Responders package for heart systemic multimodal project.

Provides tools for:
- Cross-organ receptor mapping (CellPhoneDB/CellChat-style LR analysis)
- Built-in curated ligand-receptor database
"""

from .cross_organ import CrossOrganMapper
from .lr_database import LRDatabase

__all__ = [
    "CrossOrganMapper",
    "LRDatabase",
]
