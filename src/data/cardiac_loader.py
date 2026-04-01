"""Cardiac multi-omics data loader for IRIS pipeline.

Loads real cardiac datasets from the Mac Mini and aligns them to a common gene universe:
  - bulk_rna:    GSE116250 left ventricle RPKM (57,975 genes × 65 samples)
  - proteomics:  Olink Explore 3072 (809 patients × ~981 proteins)
  - scrna:       TMS aging DE per celltype (17K genes × 9 cell types)
  - epigenomics: snRNA-seq AS vs Control DE (239 genes × 7 cell types)
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.synthetic_generator import ModalityData

logger = logging.getLogger(__name__)

CARDIAC_DATA_ROOT = Path(
    "/Users/ocm/.superset/worktrees/age related heart HF therapy/want-to-find-new-t/data"
)


def load_cardiac_data(
    max_genes: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, ModalityData], List[str], List[int], Dict[str, int]]:
    """Load and align cardiac multi-omics data.

    Returns:
        data: modality name -> ModalityData
        gene_names: ordered gene symbols in the universe
        gene_ids: sequential integer IDs
        modality_features: modality name -> feature dimension (for IrisModel)
    """
    # Load each modality
    bulk_genes, bulk_mat = _load_bulk_rna()
    prot_genes, prot_mat = _load_olink_proteomics()
    scrna_genes, scrna_mat = _load_scrna_aging_de()
    de_genes, de_mat = _load_snrna_de()

    # Build gene universe prioritizing multi-modality coverage
    gene_universe = _build_gene_universe(
        bulk_genes,
        prot_genes,
        scrna_genes,
        de_genes,
        bulk_mat,
        max_genes=max_genes,
    )
    n_genes = len(gene_universe)
    gene_ids = list(range(1, n_genes + 1))
    logger.info("Cardiac gene universe: %d genes", n_genes)

    # Align and build ModalityData
    data: Dict[str, ModalityData] = {}
    modality_features: Dict[str, int] = {}

    for name, genes, mat in [
        ("bulk_rna", bulk_genes, bulk_mat),
        ("proteomics", prot_genes, prot_mat),
        ("scrna", scrna_genes, scrna_mat),
        ("epigenomics", de_genes, de_mat),
    ]:
        feat = _align_to_universe(gene_universe, genes, mat)
        data[name] = ModalityData(
            features=feat,
            gene_ids=gene_ids,
            modality=name,
            gene_names=gene_universe,
        )
        modality_features[name] = feat.shape[1]
        coverage = 100 * (feat.abs().sum(dim=1) > 0).float().mean().item()
        logger.info(
            "  %s: %d genes × %d features, coverage %.1f%%",
            name,
            n_genes,
            feat.shape[1],
            coverage,
        )

    return data, gene_universe, gene_ids, modality_features


def _load_bulk_rna() -> Tuple[List[str], torch.Tensor]:
    """Load GSE116250 RPKM bulk RNA-seq (left ventricle, NF/DCM/ICM)."""
    path = CARDIAC_DATA_ROOT / "geo" / "GSE116250" / "GSE116250_rpkm.txt.gz"
    logger.info("Loading bulk RNA from %s", path)

    rows = []
    gene_names = []
    with gzip.open(path, "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gene_symbol = parts[1]  # Common_name column
            if not gene_symbol or gene_symbol == "":
                continue
            values = []
            for v in parts[2:]:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(0.0)
            gene_names.append(gene_symbol.upper())
            rows.append(values)

    # Deduplicate: keep first occurrence
    seen = {}
    unique_names = []
    unique_rows = []
    for name, row in zip(gene_names, rows):
        if name not in seen:
            seen[name] = True
            unique_names.append(name)
            unique_rows.append(row)

    mat = torch.tensor(unique_rows, dtype=torch.float32)
    logger.info(
        "  bulk_rna raw: %d genes × %d samples", len(unique_names), mat.shape[1]
    )
    return unique_names, mat


def _load_olink_proteomics() -> Tuple[List[str], torch.Tensor]:
    """Load Olink proteomics with gene symbols (809 patients × ~981 proteins)."""
    path = (
        CARDIAC_DATA_ROOT
        / "pressure_overload"
        / "processed"
        / "olink_proteomics_gene_symbols.csv"
    )
    logger.info("Loading Olink proteomics from %s", path)

    df = pd.read_csv(path)
    # Columns: id, age, sex, GENE1, GENE2, ...
    meta_cols = ["id", "age", "sex"]
    gene_cols = [c for c in df.columns if c not in meta_cols]

    gene_names = [c.upper() for c in gene_cols]

    # Transpose: genes as rows, patients as features
    mat = df[gene_cols].values.T  # (n_genes, n_patients)
    mat = np.nan_to_num(mat, nan=0.0)
    mat = torch.tensor(mat, dtype=torch.float32)

    logger.info(
        "  proteomics raw: %d proteins × %d patients", len(gene_names), mat.shape[1]
    )
    return gene_names, mat


def _load_scrna_aging_de() -> Tuple[List[str], torch.Tensor]:
    """Load TMS aging DE per celltype, pivoted to gene × feature matrix.

    Features per gene: logFC and -log10(pval_adj) for each cell type.
    """
    path = CARDIAC_DATA_ROOT / "scrna_aging" / "processed" / "aging_de_per_celltype.csv"
    logger.info("Loading scRNA aging DE from %s", path)

    df = pd.read_csv(path)

    # Focus on cardiac-relevant tissues
    cardiac_celltypes = df["cell_type"].unique().tolist()

    features_list = []
    feature_names = []
    for ct in cardiac_celltypes:
        ct_df = df[df["cell_type"] == ct][
            ["names", "logfoldchanges", "pvals_adj"]
        ].copy()
        ct_short = ct.replace(" ", "_")[:15]
        ct_df = ct_df.rename(
            columns={
                "logfoldchanges": f"logFC_{ct_short}",
                "pvals_adj": f"padj_{ct_short}",
            }
        )
        ct_df[f"nlogp_{ct_short}"] = -np.log10(
            ct_df[f"padj_{ct_short}"].clip(lower=1e-300)
        )
        ct_df = ct_df.drop(columns=[f"padj_{ct_short}"])
        features_list.append(ct_df.set_index("names"))
        feature_names.extend([f"logFC_{ct_short}", f"nlogp_{ct_short}"])

    # Merge all cell type features
    merged = features_list[0]
    for feat_df in features_list[1:]:
        merged = merged.join(feat_df, how="outer")

    merged = merged.fillna(0.0)
    gene_names = [g.upper() for g in merged.index.tolist()]
    mat = torch.tensor(merged.values, dtype=torch.float32)

    logger.info(
        "  scrna raw: %d genes × %d features (%d cell types)",
        len(gene_names),
        mat.shape[1],
        len(cardiac_celltypes),
    )
    return gene_names, mat


def _load_pseudobulk_from_h5ad(
    h5ad_path: Path,
    groupby: str = "cell_ontology_class",
    age_col: str = "age",
) -> Tuple[List[str], torch.Tensor]:
    """Extract pseudobulk from TMS heart h5ad: gene × (celltype × age_group) features.

    For each cell type:
      - logFC: mean expression in old (18m, 24m) / young (3m)
      - nlogp: -log10(p-value) from t-test

    Returns:
        gene_names: List of gene symbols
        features: Tensor of shape (n_genes, n_celltypes * 2)
    """
    import scanpy as sc
    from scipy import stats

    logger.info("Loading pseudobulk from h5ad: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)

    # Simplify: young (3m) vs old (18m, 24m)
    adata.obs["age_group"] = adata.obs[age_col].apply(
        lambda x: "old" if x in ["18m", "24m"] else "young"
    )

    celltypes = adata.obs[groupby].unique()
    features_dict = {}

    for ct in celltypes:
        ct_data = adata[adata.obs[groupby] == ct]

        old_data = ct_data[ct_data.obs["age_group"] == "old"]
        young_data = ct_data[ct_data.obs["age_group"] == "young"]

        if old_data.n_obs < 5 or young_data.n_obs < 5:
            logger.warning(
                f"  Skipping {ct}: insufficient cells (old={old_data.n_obs}, young={young_data.n_obs})"
            )
            continue

        # Mean expression per group
        old_X = old_data.X.toarray() if hasattr(old_data.X, "toarray") else old_data.X
        young_X = (
            young_data.X.toarray() if hasattr(young_data.X, "toarray") else young_data.X
        )

        old_mean = np.asarray(old_X).mean(axis=0).flatten()
        young_mean = np.asarray(young_X).mean(axis=0).flatten()

        # logFC (old vs young) - clip to avoid inf
        logfc = np.log2(np.clip(old_mean + 1, 1e-10, None)) - np.log2(
            np.clip(young_mean + 1, 1e-10, None)
        )
        logfc = np.nan_to_num(logfc, nan=0.0, posinf=10.0, neginf=-10.0)

        # t-test p-values
        try:
            _, pvals = stats.ttest_ind(old_X, young_X, equal_var=False)
            pvals = np.asarray(pvals).flatten()
            pvals = np.nan_to_num(pvals, nan=1.0)  # Handle NaN
            pvals = np.clip(pvals, 1e-300, 1.0)  # Avoid log(0)
            nlogp = -np.log10(pvals)
            nlogp = np.clip(nlogp, 0, 300)  # Cap extreme values
        except Exception:
            nlogp = np.zeros_like(logfc)

        ct_short = ct.replace(" ", "_")[:10]
        features_dict[f"logFC_{ct_short}"] = logfc
        features_dict[f"nlogp_{ct_short}"] = nlogp

    # Stack features: (n_genes, n_celltypes * 2)
    feature_names = sorted(features_dict.keys())
    features = np.column_stack([features_dict[k] for k in feature_names])

    gene_names = [g.upper() for g in adata.var_names.tolist()]

    n_celltypes = len(features_dict) // 2
    coverage = 100 * (np.abs(features).sum(axis=1) > 0).mean()
    logger.info(
        "  Pseudobulk: %d genes × %d features (%d cell types), coverage %.1f%%",
        len(gene_names),
        features.shape[1],
        n_celltypes,
        coverage,
    )

    return gene_names, torch.tensor(features, dtype=torch.float32)


def _load_snrna_de() -> Tuple[List[str], torch.Tensor]:
    """Load snRNA-seq pseudobulk features.

    Tries h5ad first (high coverage ~30%), falls back to CSV if unavailable.
    """
    # Try h5ad first for better coverage
    h5ad_path = (
        CARDIAC_DATA_ROOT / "scrna_aging" / "processed" / "tms_heart_muscle_qc.h5ad"
    )

    if h5ad_path.exists():
        try:
            return _load_pseudobulk_from_h5ad(h5ad_path)
        except Exception as e:
            logger.warning("Failed to load h5ad: %s, falling back to CSV", e)

    # Fallback to CSV
    path = (
        CARDIAC_DATA_ROOT
        / "pressure_overload"
        / "results"
        / "snrnaseq_de_AS_vs_Control.csv"
    )
    logger.info("Loading snRNA DE (AS vs Control) from %s", path)

    df = pd.read_csv(path)

    celltypes = df["cell_type"].unique().tolist()
    features_list = []

    for ct in celltypes:
        ct_df = df[df["cell_type"] == ct][
            ["gene", "logFC_AS_vs_Ctrl", "FDR", "mean_AS", "mean_Ctrl"]
        ].copy()
        ct_short = ct[:8]
        ct_df[f"nlogFDR_{ct_short}"] = -np.log10(ct_df["FDR"].clip(lower=1e-300))
        ct_df = ct_df.rename(
            columns={
                "logFC_AS_vs_Ctrl": f"logFC_{ct_short}",
                "mean_AS": f"mAS_{ct_short}",
                "mean_Ctrl": f"mCtrl_{ct_short}",
            }
        )
        ct_df = ct_df.drop(columns=["FDR"])
        ct_df = ct_df.set_index("gene")
        features_list.append(ct_df)

    merged = features_list[0]
    for feat_df in features_list[1:]:
        merged = merged.join(feat_df, how="outer", rsuffix="_dup")

    # Remove any duplicate columns
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.fillna(0.0)

    gene_names = [g.upper() for g in merged.index.tolist()]
    mat = torch.tensor(merged.values, dtype=torch.float32)

    logger.info(
        "  snRNA DE raw: %d genes × %d features (%d cell types)",
        len(gene_names),
        mat.shape[1],
        len(celltypes),
    )
    return gene_names, mat


def _build_gene_universe(
    bulk_genes: List[str],
    prot_genes: List[str],
    scrna_genes: List[str],
    de_genes: List[str],
    bulk_mat: torch.Tensor,
    max_genes: int = 2000,
) -> List[str]:
    """Build gene universe prioritizing multi-modality and high-variance genes."""
    # Count how many modalities each gene appears in
    all_gene_sets = [set(bulk_genes), set(prot_genes), set(scrna_genes), set(de_genes)]
    gene_counts: Dict[str, int] = {}
    for gs in all_gene_sets:
        for g in gs:
            gene_counts[g] = gene_counts.get(g, 0) + 1

    # Compute variance for bulk RNA genes
    gene_var: Dict[str, float] = {}
    for i, g in enumerate(bulk_genes):
        if i < bulk_mat.shape[0]:
            gene_var[g] = bulk_mat[i].var().item()

    # Score: modality_count * 1000 + variance_rank
    # First: all genes in proteomics (ensures good coverage)
    selected = set()

    # Priority 1: genes in 3+ modalities
    for g, count in gene_counts.items():
        if count >= 3:
            selected.add(g)

    # Priority 2: all Olink proteins (these are disease-relevant)
    selected.update(prot_genes)

    # Priority 3: genes in 2+ modalities
    for g, count in gene_counts.items():
        if count >= 2 and len(selected) < max_genes:
            selected.add(g)

    # Priority 4: top variable bulk RNA genes
    if len(selected) < max_genes:
        remaining = [(g, v) for g, v in gene_var.items() if g not in selected]
        remaining.sort(key=lambda x: x[1], reverse=True)
        for g, _ in remaining:
            if len(selected) >= max_genes:
                break
            selected.add(g)

    # Sort for reproducibility
    result = sorted(selected)[:max_genes]

    n_multi = sum(1 for g in result if gene_counts.get(g, 0) >= 2)
    n_prot = sum(1 for g in result if g in set(prot_genes))
    logger.info(
        "Gene universe: %d total, %d in 2+ modalities, %d with proteomics",
        len(result),
        n_multi,
        n_prot,
    )
    return result


def _align_to_universe(
    universe: List[str],
    source_genes: List[str],
    source_mat: torch.Tensor,
) -> torch.Tensor:
    """Align source data to gene universe, zero-padding missing genes."""
    n_genes = len(universe)
    n_features = source_mat.shape[1]

    # Build lookup
    source_lookup = {}
    for i, g in enumerate(source_genes):
        if g not in source_lookup:  # keep first occurrence
            source_lookup[g] = i

    aligned = torch.zeros(n_genes, n_features)
    for i, gene in enumerate(universe):
        src_idx = source_lookup.get(gene)
        if src_idx is not None and src_idx < source_mat.shape[0]:
            aligned[i] = source_mat[src_idx]

    return aligned
