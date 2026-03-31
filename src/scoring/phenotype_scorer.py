"""
Cardiac phenotype scoring using gene set enrichment on single-cell data.

Uses scanpy.tl.score_genes() to compute per-cell phenotype scores for
seven cardiac pathological and physiological programs. Supports per-cluster
aggregation, dotplot, and heatmap visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated cardiac gene sets
# ---------------------------------------------------------------------------

CARDIAC_GENE_SETS: Dict[str, List[str]] = {
    "fibrosis": [
        "COL1A1", "COL3A1", "FN1", "ACTA2", "TGFB1",
        "CTGF", "LOX", "POSTN",
    ],
    "ECM_remodeling": [
        "MMP1", "MMP2", "MMP3", "MMP9", "MMP14",
        "TIMP1", "TIMP2", "ADAMTS2",
    ],
    "inflammation": [
        "TNF", "IL1B", "IL6", "CXCL8", "CCL2",
        "CCL3", "NLRP3", "NFkBIA",
    ],
    "hypertrophy": [
        "NPPA", "NPPB", "MYH7", "ACTA1", "ANKRD1", "XIRP2",
    ],
    "mito_stress": [
        "PINK1", "PARK2", "BNIP3", "BNIP3L",
        "SOD2", "CAT", "GPX1",
    ],
    "senescence": [
        "CDKN1A", "CDKN2A", "TP53", "SERPINE1", "IGFBP3", "GLB1",
    ],
    "secretory": [
        "VEGFA", "FGF2", "PDGFA", "IGF1", "BMP4",
        "WNT5A", "DKK1", "SFRP1",
    ],
}

SCORE_COLUMN_PREFIX = "score_"


class CardiacPhenotypeScorer:
    """Score cardiac phenotypes per cell using curated gene sets.

    Parameters
    ----------
    adata:
        AnnData object with raw or normalised counts.
    gene_sets:
        Mapping of phenotype name to gene list. Defaults to
        ``CARDIAC_GENE_SETS``.
    cluster_key:
        ``adata.obs`` column used for cluster-level aggregation and
        visualisations (e.g. ``"cell_type"`` or ``"leiden"``).
    figures_dir:
        Directory where figures are saved. Created if absent.
    results_dir:
        Directory where CSV outputs are saved. Created if absent.
    n_bins:
        Number of expression bins passed to ``score_genes`` (default 25).
    ctrl_size:
        Number of control genes per gene in the set (default 50).
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        adata: AnnData,
        gene_sets: Optional[Dict[str, List[str]]] = None,
        cluster_key: str = "cell_type",
        figures_dir: Path = Path("results/figures"),
        results_dir: Path = Path("results"),
        n_bins: int = 25,
        ctrl_size: int = 50,
        random_state: int = 42,
    ) -> None:
        self.adata = adata
        self.gene_sets: Dict[str, List[str]] = gene_sets if gene_sets is not None else CARDIAC_GENE_SETS
        self.cluster_key = cluster_key
        self.figures_dir = Path(figures_dir)
        self.results_dir = Path(results_dir)
        self.n_bins = n_bins
        self.ctrl_size = ctrl_size
        self.random_state = random_state

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._score_columns: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, save_tables: bool = True) -> AnnData:
        """Run all phenotype scores and optionally save aggregated tables.

        Parameters
        ----------
        save_tables:
            If ``True``, write per-cluster mean-score CSV to
            ``results_dir/cluster_phenotype_scores.csv``.

        Returns
        -------
        AnnData
            The input ``adata`` with new ``.obs`` columns for each score.
        """
        logger.info("Starting cardiac phenotype scoring (%d cells).", self.adata.n_obs)
        self._score_all_phenotypes()
        if save_tables:
            self._save_cluster_scores()
        logger.info("Phenotype scoring complete.")
        return self.adata

    def plot_dotplot(self, groupby: Optional[str] = None) -> Path:
        """Generate a dotplot of mean phenotype scores across clusters.

        Parameters
        ----------
        groupby:
            ``adata.obs`` key to group cells. Defaults to ``self.cluster_key``.

        Returns
        -------
        Path
            Path to the saved PNG figure.
        """
        groupby = groupby or self.cluster_key
        if not self._score_columns:
            raise RuntimeError("Call run() before plot_dotplot().")
        if groupby not in self.adata.obs.columns:
            raise KeyError(f"groupby key '{groupby}' not in adata.obs.")

        fig_path = self.figures_dir / "phenotype_dotplot.png"
        logger.info("Generating dotplot grouped by '%s'.", groupby)

        # Build a DataFrame: rows = clusters, cols = phenotypes
        cluster_means = self._compute_cluster_means(groupby)
        cluster_frac = self._compute_positive_fraction(groupby)

        n_pheno = len(self._score_columns)
        n_clust = len(cluster_means)
        phenotype_labels = [c.replace(SCORE_COLUMN_PREFIX, "") for c in self._score_columns]

        fig, ax = plt.subplots(figsize=(max(6, n_pheno * 1.2), max(4, n_clust * 0.8)))

        for ci, cluster in enumerate(cluster_means.index):
            for pi, col in enumerate(self._score_columns):
                mean_val = cluster_means.loc[cluster, col]
                frac_val = cluster_frac.loc[cluster, col]
                size = max(10, frac_val * 300)
                color_val = np.clip(mean_val, -2, 2)
                ax.scatter(
                    pi,
                    ci,
                    s=size,
                    c=[[plt.cm.RdBu_r((color_val + 2) / 4)]],  # type: ignore[attr-defined]
                    edgecolors="grey",
                    linewidths=0.4,
                )

        ax.set_xticks(range(n_pheno))
        ax.set_xticklabels(phenotype_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_clust))
        ax.set_yticklabels(cluster_means.index, fontsize=9)
        ax.set_xlabel("Phenotype")
        ax.set_ylabel(groupby)
        ax.set_title("Cardiac Phenotype Scores (dot size = fraction positive)")

        # Colourbar
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-2, vmax=2))  # type: ignore[attr-defined]
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Mean score (clipped ±2)")

        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Dotplot saved to %s.", fig_path)
        return fig_path

    def plot_heatmap(self, groupby: Optional[str] = None) -> Path:
        """Generate a heatmap of mean phenotype scores across clusters.

        Parameters
        ----------
        groupby:
            ``adata.obs`` key to group cells. Defaults to ``self.cluster_key``.

        Returns
        -------
        Path
            Path to the saved PNG figure.
        """
        groupby = groupby or self.cluster_key
        if not self._score_columns:
            raise RuntimeError("Call run() before plot_heatmap().")

        fig_path = self.figures_dir / "phenotype_heatmap.png"
        logger.info("Generating heatmap grouped by '%s'.", groupby)

        cluster_means = self._compute_cluster_means(groupby)
        phenotype_labels = [c.replace(SCORE_COLUMN_PREFIX, "") for c in self._score_columns]
        cluster_means.columns = phenotype_labels  # type: ignore[assignment]

        n_clust, n_pheno = cluster_means.shape
        fig, ax = plt.subplots(figsize=(max(6, n_pheno * 1.1), max(3, n_clust * 0.7)))

        mat = cluster_means.values.astype(float)
        # Z-score across clusters for each phenotype
        std = mat.std(axis=0)
        std[std == 0] = 1.0
        mat_z = (mat - mat.mean(axis=0)) / std

        im = ax.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(n_pheno))
        ax.set_xticklabels(phenotype_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_clust))
        ax.set_yticklabels(cluster_means.index, fontsize=9)
        ax.set_title("Cardiac Phenotype Scores — Z-scored per phenotype")
        fig.colorbar(im, ax=ax, label="Z-score")

        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap saved to %s.", fig_path)
        return fig_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_all_phenotypes(self) -> None:
        """Iterate over gene sets and call score_genes for each."""
        var_names: set = set(self.adata.var_names)
        for phenotype, genes in self.gene_sets.items():
            obs_col = f"{SCORE_COLUMN_PREFIX}{phenotype}"
            available = [g for g in genes if g in var_names]
            missing = set(genes) - set(available)
            if missing:
                logger.warning(
                    "Phenotype '%s': %d/%d genes missing from adata: %s",
                    phenotype,
                    len(missing),
                    len(genes),
                    sorted(missing),
                )
            if len(available) < 2:
                logger.warning(
                    "Phenotype '%s': fewer than 2 genes available (%d). Skipping.",
                    phenotype,
                    len(available),
                )
                self.adata.obs[obs_col] = np.nan
                self._score_columns.append(obs_col)
                continue

            logger.info(
                "Scoring phenotype '%s' with %d genes.", phenotype, len(available)
            )
            sc.tl.score_genes(
                self.adata,
                gene_list=available,
                score_name=obs_col,
                n_bins=self.n_bins,
                ctrl_size=min(self.ctrl_size, self.adata.n_vars - len(available) - 1),
                random_state=self.random_state,
            )
            self._score_columns.append(obs_col)
            logger.debug("Stored score '%s' in adata.obs.", obs_col)

    def _compute_cluster_means(self, groupby: str) -> pd.DataFrame:
        """Return DataFrame of mean scores per cluster."""
        df = self.adata.obs[[groupby] + self._score_columns].copy()
        return df.groupby(groupby)[self._score_columns].mean()

    def _compute_positive_fraction(self, groupby: str, threshold: float = 0.0) -> pd.DataFrame:
        """Return DataFrame of fraction of cells with score > threshold per cluster."""
        df = self.adata.obs[[groupby] + self._score_columns].copy()
        above = df[self._score_columns].gt(threshold).astype(float)
        above[groupby] = df[groupby]
        return above.groupby(groupby)[self._score_columns].mean()

    def _save_cluster_scores(self) -> None:
        """Write per-cluster mean scores to CSV."""
        if self.cluster_key not in self.adata.obs.columns:
            logger.warning(
                "cluster_key '%s' not in adata.obs — skipping CSV export.",
                self.cluster_key,
            )
            return
        out_path = self.results_dir / "cluster_phenotype_scores.csv"
        cluster_means = self._compute_cluster_means(self.cluster_key)
        cluster_means.columns = [c.replace(SCORE_COLUMN_PREFIX, "") for c in cluster_means.columns]
        cluster_means.to_csv(out_path)
        logger.info("Cluster phenotype scores saved to %s.", out_path)

    # ------------------------------------------------------------------
    # Convenience: per-cell score DataFrame
    # ------------------------------------------------------------------

    @property
    def score_df(self) -> pd.DataFrame:
        """Return a DataFrame of per-cell phenotype scores.

        Raises
        ------
        RuntimeError
            If ``run()`` has not been called yet.
        """
        if not self._score_columns:
            raise RuntimeError("Call run() first.")
        df = self.adata.obs[self._score_columns].copy()
        df.columns = [c.replace(SCORE_COLUMN_PREFIX, "") for c in df.columns]
        return df
