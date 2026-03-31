"""
Utility modules for the Heart-Systemic Multi-modal Pipeline.

Provides:
    - config_loader: YAML config loading and validation
    - logger: Rich-based structured logging
    - memory: RAM-aware downsampling and caching helpers
    - dataset_registry: Curated dataset catalogue loader and query interface
    - gene_sets: Curated gene sets for module scoring and annotation
"""

from src.utils.config_loader import load_config, ConfigLoader
from src.utils.logger import get_logger, setup_logging
from src.utils.memory import (
    check_available_ram,
    downsample_adata,
    cached_load,
)
from src.utils.dataset_registry import DatasetRegistry
from src.utils.gene_sets import (
    get_gene_set,
    list_gene_sets,
    get_all_sets,
    gene_set_sizes,
    genes_in_set,
)

__all__ = [
    "load_config",
    "ConfigLoader",
    "get_logger",
    "setup_logging",
    "check_available_ram",
    "downsample_adata",
    "cached_load",
    "DatasetRegistry",
    "get_gene_set",
    "list_gene_sets",
    "get_all_sets",
    "gene_set_sizes",
    "genes_in_set",
]
