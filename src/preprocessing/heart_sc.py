"""
Heart single-cell RNA-seq preprocessing pipeline.

Loads raw h5ad or MTX data, applies QC filtering, normalization,
dimensionality reduction, clustering, and marker-based cell type annotation.
Results are cached to disk as a processed AnnData object.
"""

import gc
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell type marker genes (heart-relevant)
# ---------------------------------------------------------------------------
HEART_CELL_MARKERS: Dict[str, List[str]] = {
    "Cardiomyocyte": ["TNNT2", "MYH7", "MYH6", "ACTC1", "TNNC1", "TTN", "RYR2"],
    "Fibroblast": ["DCN", "LUM", "COL1A1", "COL3A1", "THY1", "POSTN", "VIM"],
    "Endothelial": ["PECAM1", "CDH5", "VWF", "CLDN5", "PLVAP", "KDR", "ENG"],
    "Smooth_Muscle": ["ACTA2", "TAGLN", "MYH11", "CNN1", "CALD1", "SMTN"],
    "Macrophage": ["CD68", "CD163", "MRC1", "CSF1R", "FCGR3A", "MSR1"],
    "T_Cell": ["CD3D", "CD3E", "CD3G", "CD8A", "CD4", "TRAC", "IL7R"],
    "NK_Cell": ["NKG7", "GNLY", "KLRD1", "NCAM1", "PRF1"],
    "B_Cell": ["CD79A", "CD79B", "MS4A1", "CD19", "IGHM"],
    "Pericyte": ["RGS5", "PDGFRB", "ABCC9", "KCNJ8", "NOTCH3"],
    "Adipocyte": ["ADIPOQ", "FABP4", "PLIN1", "LEP", "PPARG"],
    "Epicardial": ["WT1", "TBX18", "ALDH1A2", "MSLN", "UPK1B"],
    "Neuronal": ["NRXN1", "NRXN3", "SLIT2", "S100B", "PLP1"],
}


class HeartSCPreprocessor:
    """
    End-to-end single-cell preprocessing for heart datasets.

    Pipeline
    --------
    1. Load h5ad / MTX / CSV
    2. QC metrics and filtering
    3. Normalization and log1p
    4. Highly variable gene (HVG) selection
    5. PCA
    6. Neighbors graph
    7. UMAP
    8. Leiden clustering
    9. Marker-based cell type annotation
    10. Save processed AnnData to cache

    Usage
    -----
    >>> pp = HeartSCPreprocessor(cache_dir=Path("data/processed"))
    >>> adata = pp.run(Path("data/raw/geo/GSE183852/sample.h5ad"))
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/processed/sc"),
        min_genes: int = 200,
        min_cells: int = 3,
        max_mito_pct: float = 20.0,
        n_top_genes: int = 3000,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        leiden_resolution: float = 0.5,
        max_cells: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        cache_dir:
            Directory for saving processed AnnData objects.
        min_genes:
            Minimum number of genes per cell.
        min_cells:
            Minimum number of cells per gene.
        max_mito_pct:
            Maximum mitochondrial gene percentage per cell.
        n_top_genes:
            Number of highly variable genes to select.
        n_pcs:
            Number of PCA components.
        n_neighbors:
            Number of neighbors for graph construction.
        leiden_resolution:
            Resolution parameter for Leiden clustering.
        max_cells:
            If set, downsample to this number of cells (memory-aware).
        random_state:
            Random seed for reproducibility.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.min_genes = min_genes
        self.min_cells = min_cells
        self.max_mito_pct = max_mito_pct
        self.n_top_genes = n_top_genes
        self.n_pcs = n_pcs
        self.n_neighbors = n_neighbors
        self.leiden_resolution = leiden_resolution
        self.max_cells = max_cells
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: Union[Path, str],
        sample_name: Optional[str] = None,
        force_recompute: bool = False,
    ):
        """
        Run the full preprocessing pipeline on a single dataset.

        Parameters
        ----------
        input_path:
            Path to h5ad, MTX directory, or CSV/TSV file.
        sample_name:
            Name for cache key. Defaults to input filename stem.
        force_recompute:
            If True, ignore cached results.

        Returns
        -------
        Processed AnnData object.
        """
        import scanpy as sc

        input_path = Path(input_path)
        sample_name = sample_name or input_path.stem
        cache_path = self.cache_dir / f"{sample_name}_processed.h5ad"

        if cache_path.exists() and not force_recompute:
            logger.info("Loading cached result: %s", cache_path)
            return sc.read_h5ad(cache_path)

        logger.info("Starting preprocessing for: %s", sample_name)

        adata = self._load(input_path)
        logger.info("Loaded: %d cells x %d genes", adata.n_obs, adata.n_vars)

        adata = self._qc_filter(adata)
        logger.info("After QC: %d cells x %d genes", adata.n_obs, adata.n_vars)

        if self.max_cells and adata.n_obs > self.max_cells:
            adata = self._downsample(adata)
            logger.info("Downsampled to: %d cells", adata.n_obs)

        adata = self._normalize(adata)
        adata = self._select_hvgs(adata)
        adata = self._pca(adata)
        adata = self._neighbors_umap(adata)
        adata = self._cluster(adata)
        adata = self._annotate_cell_types(adata)

        adata.write_h5ad(cache_path)
        logger.info("Saved processed AnnData to: %s", cache_path)

        return adata

    def run_batch(
        self,
        input_paths: List[Union[Path, str]],
        concat_key: str = "dataset",
        force_recompute: bool = False,
    ):
        """
        Preprocess multiple datasets and concatenate them.

        Parameters
        ----------
        input_paths:
            List of paths to h5ad / MTX files.
        concat_key:
            obs column name for batch origin.
        force_recompute:
            Recompute even if cached.

        Returns
        -------
        Concatenated AnnData.
        """
        import scanpy as sc

        adatas = []
        for path in input_paths:
            path = Path(path)
            try:
                adata = self.run(path, force_recompute=force_recompute)
                adata.obs[concat_key] = path.stem
                adatas.append(adata)
            except Exception as exc:
                logger.error("Failed to process %s: %s", path, exc)

        if not adatas:
            raise ValueError("No datasets successfully preprocessed")

        logger.info("Concatenating %d datasets", len(adatas))
        combined = sc.concat(adatas, label=concat_key, keys=[Path(p).stem for p in input_paths])

        # Recompute embedding on combined data
        sc.pp.highly_variable_genes(combined, n_top_genes=self.n_top_genes, batch_key=concat_key)
        sc.tl.pca(combined, n_comps=self.n_pcs, random_state=self.random_state)
        sc.pp.neighbors(combined, n_neighbors=self.n_neighbors, random_state=self.random_state)
        sc.tl.umap(combined, random_state=self.random_state)
        sc.tl.leiden(combined, resolution=self.leiden_resolution, random_state=self.random_state)

        cache_path = self.cache_dir / "combined_processed.h5ad"
        combined.write_h5ad(cache_path)
        logger.info("Saved combined AnnData to: %s", cache_path)

        return combined

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _load(self, path: Path):
        """
        Load single-cell data from h5ad, MTX directory, or CSV/TSV.

        Parameters
        ----------
        path:
            Input path (file or directory for MTX).

        Returns
        -------
        AnnData object.
        """
        import scanpy as sc

        suffix = path.suffix.lower()

        if suffix == ".h5ad":
            logger.info("Reading h5ad: %s", path)
            return sc.read_h5ad(path)

        elif suffix in (".h5", ".hdf5"):
            logger.info("Reading HDF5: %s", path)
            try:
                return sc.read_10x_h5(path)
            except Exception:
                return sc.read_hdf(path, key="matrix")

        elif path.is_dir() or suffix == ".mtx" or suffix == ".mtx.gz":
            mtx_dir = path if path.is_dir() else path.parent
            logger.info("Reading MTX directory: %s", mtx_dir)
            return sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=True)

        elif suffix in (".csv",):
            logger.info("Reading CSV: %s", path)
            df = pd.read_csv(path, index_col=0)
            import anndata
            return anndata.AnnData(X=df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))

        elif suffix in (".tsv", ".txt"):
            logger.info("Reading TSV: %s", path)
            df = pd.read_csv(path, sep="\t", index_col=0)
            import anndata
            return anndata.AnnData(X=df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=df.columns))

        else:
            raise ValueError(f"Unsupported file format: {suffix} for {path}")

    def _qc_filter(self, adata):
        """
        Compute QC metrics and filter cells/genes.

        Filters on:
        - Minimum genes per cell
        - Minimum cells per gene
        - Mitochondrial gene percentage

        Returns filtered AnnData.
        """
        import scanpy as sc

        # Identify mitochondrial genes
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        mito_count = adata.var["mt"].sum()
        if mito_count == 0:
            # Try lowercase
            adata.var["mt"] = adata.var_names.str.startswith("mt-")
            logger.debug("Using lowercase mt- prefix (%d genes)", adata.var["mt"].sum())

        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

        n_before = adata.n_obs
        sc.pp.filter_cells(adata, min_genes=self.min_genes)
        sc.pp.filter_genes(adata, min_cells=self.min_cells)

        # Mito filter
        adata = adata[adata.obs["pct_counts_mt"] < self.max_mito_pct].copy()

        logger.info(
            "QC: removed %d low-quality cells (min_genes=%d, max_mito=%.0f%%)",
            n_before - adata.n_obs,
            self.min_genes,
            self.max_mito_pct,
        )
        return adata

    def _downsample(self, adata):
        """
        Randomly downsample cells to max_cells for memory efficiency.

        Returns downsampled AnnData.
        """
        import scanpy as sc

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(adata.n_obs, size=self.max_cells, replace=False)
        idx.sort()
        logger.info("Downsampling %d -> %d cells", adata.n_obs, self.max_cells)
        return adata[idx].copy()

    def _normalize(self, adata):
        """
        Normalize counts and log-transform.

        Steps:
        1. Library-size normalization to 10,000 counts per cell
        2. log1p transform
        3. Save raw counts in adata.raw
        """
        import scanpy as sc

        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.raw = adata
        logger.info("Normalized and log-transformed expression matrix")
        return adata

    def _select_hvgs(self, adata):
        """
        Select highly variable genes using Seurat flavor.

        Returns AnnData with HVG annotation in adata.var.
        """
        import scanpy as sc

        n_top = min(self.n_top_genes, adata.n_vars)
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top,
            flavor="seurat_v3",
            layer="counts",
        )
        n_hvg = adata.var["highly_variable"].sum()
        logger.info("Selected %d highly variable genes", n_hvg)

        # Scale only HVGs for PCA
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(adata_hvg, max_value=10)
        adata.obsm["X_scaled_hvg"] = adata_hvg.X
        return adata

    def _pca(self, adata):
        """
        Run PCA on the scaled HVG matrix.

        Returns AnnData with adata.obsm['X_pca'].
        """
        import scanpy as sc
        from scipy.sparse import issparse

        n_pcs = min(self.n_pcs, adata.n_obs - 1, adata.n_vars - 1)

        if "X_scaled_hvg" in adata.obsm:
            # Use precomputed scaled HVG matrix
            X_in = adata.obsm["X_scaled_hvg"]
            if issparse(X_in):
                X_in = X_in.toarray()

            from sklearn.decomposition import PCA as SkPCA
            pca_model = SkPCA(n_components=n_pcs, random_state=self.random_state)
            adata.obsm["X_pca"] = pca_model.fit_transform(X_in)
            adata.uns["pca"] = {"variance_ratio": pca_model.explained_variance_ratio_}
        else:
            sc.tl.pca(adata, n_comps=n_pcs, random_state=self.random_state)

        logger.info("PCA: %d components", n_pcs)
        return adata

    def _neighbors_umap(self, adata):
        """
        Build the k-NN graph and compute UMAP embedding.

        Returns AnnData with adata.obsm['X_umap'].
        """
        import scanpy as sc

        sc.pp.neighbors(
            adata,
            n_neighbors=self.n_neighbors,
            n_pcs=min(self.n_pcs, adata.obsm["X_pca"].shape[1]),
            random_state=self.random_state,
        )
        sc.tl.umap(adata, random_state=self.random_state)
        logger.info("Computed UMAP embedding")
        return adata

    def _cluster(self, adata):
        """
        Cluster cells using the Leiden algorithm.

        Returns AnnData with adata.obs['leiden'] cluster labels.
        """
        import scanpy as sc

        sc.tl.leiden(
            adata,
            resolution=self.leiden_resolution,
            random_state=self.random_state,
        )
        n_clusters = adata.obs["leiden"].nunique()
        logger.info("Leiden clustering: %d clusters (resolution=%.2f)", n_clusters, self.leiden_resolution)
        return adata

    def _annotate_cell_types(self, adata):
        """
        Annotate cell types using predefined marker gene scores.

        For each cell, computes a score for each cell type based on
        the mean expression of its marker genes. Assigns the highest-
        scoring cell type as the annotation.

        Adds ``adata.obs['cell_type']`` and per-type score columns.

        Returns annotated AnnData.
        """
        import scanpy as sc

        var_names_upper = adata.var_names.str.upper()
        scores: Dict[str, np.ndarray] = {}

        for cell_type, markers in HEART_CELL_MARKERS.items():
            # Find which markers are present in this dataset
            present = [m for m in markers if m.upper() in var_names_upper]
            if not present:
                logger.debug("No markers found for cell type: %s", cell_type)
                scores[cell_type] = np.zeros(adata.n_obs)
                continue

            # Use scanpy score_genes for robustness
            try:
                sc.tl.score_genes(
                    adata,
                    gene_list=present,
                    score_name=f"score_{cell_type}",
                    random_state=self.random_state,
                )
                scores[cell_type] = adata.obs[f"score_{cell_type}"].values
            except Exception as exc:
                logger.warning("score_genes failed for %s: %s", cell_type, exc)
                scores[cell_type] = np.zeros(adata.n_obs)

        # Assign cell type with maximum score
        score_df = pd.DataFrame(scores, index=adata.obs_names)
        adata.obs["cell_type"] = score_df.idxmax(axis=1)

        # Flag low-confidence assignments (all scores near zero)
        max_scores = score_df.max(axis=1)
        threshold = max_scores.quantile(0.10)
        adata.obs.loc[max_scores < threshold, "cell_type"] = "Unknown"

        counts = adata.obs["cell_type"].value_counts()
        logger.info("Cell type annotation summary:\n%s", counts.to_string())

        return adata

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def compute_cluster_markers(self, adata, groupby: str = "leiden", n_genes: int = 25):
        """
        Compute differentially expressed marker genes per cluster.

        Parameters
        ----------
        adata:
            Processed AnnData.
        groupby:
            obs column to group by.
        n_genes:
            Number of top marker genes per group.

        Returns
        -------
        DataFrame with marker gene results.
        """
        import scanpy as sc

        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            method="wilcoxon",
            n_genes=n_genes,
            use_raw=True,
        )
        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names

        rows = []
        for group in groups:
            for rank in range(n_genes):
                rows.append({
                    "cluster": group,
                    "rank": rank + 1,
                    "gene": result["names"][group][rank],
                    "score": result["scores"][group][rank],
                    "logfoldchange": result["logfoldchanges"][group][rank],
                    "pval_adj": result["pvals_adj"][group][rank],
                })

        return pd.DataFrame(rows)

    def plot_qc(self, adata, output_dir: Optional[Path] = None) -> None:
        """
        Generate QC violin plots.

        Parameters
        ----------
        adata:
            AnnData (post-QC metrics calculation).
        output_dir:
            Directory to save plots. If None, displays interactively.
        """
        try:
            import scanpy as sc
            import matplotlib
            if output_dir:
                matplotlib.use("Agg")

            sc.pl.violin(
                adata,
                ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                jitter=0.4,
                show=output_dir is None,
                save=str(output_dir / "qc_violin.pdf") if output_dir else None,
            )
        except Exception as exc:
            logger.warning("QC plot failed: %s", exc)

    def get_summary(self, adata) -> Dict:
        """
        Return a summary dict of the processed AnnData.

        Returns dict with n_cells, n_genes, n_clusters, cell_type_counts.
        """
        summary = {
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_clusters": adata.obs["leiden"].nunique() if "leiden" in adata.obs else None,
            "has_umap": "X_umap" in adata.obsm,
            "has_cell_type": "cell_type" in adata.obs,
        }
        if "cell_type" in adata.obs:
            summary["cell_type_counts"] = adata.obs["cell_type"].value_counts().to_dict()
        return summary
