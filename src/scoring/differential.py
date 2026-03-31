"""
Differential expression analysis for cardiac single-cell data.

Performs disease-vs-control comparisons per cell type using Wilcoxon
rank-sum tests via scanpy. Outputs DEG tables and volcano plots.
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

# Column name conventions for the DEG table
_LOG2FC_COL = "log2fc"
_PVAL_COL = "pval"
_PADJ_COL = "padj"
_SCORE_COL = "wilcoxon_score"
_GENE_COL = "gene"
_CELL_TYPE_COL = "cell_type"


class DifferentialExpression:
    """Disease-vs-control differential expression per cell type.

    Parameters
    ----------
    adata:
        AnnData with raw / log-normalised counts. Must contain
        ``condition_key`` and ``cell_type_key`` in ``.obs``.
    condition_key:
        ``adata.obs`` column distinguishing disease from control
        (e.g. ``"condition"``).
    disease_label:
        Value in ``condition_key`` that marks disease cells.
    control_label:
        Value in ``condition_key`` that marks control cells.
    cell_type_key:
        ``adata.obs`` column for cell type labels.
    figures_dir:
        Directory for saved figures.
    results_dir:
        Directory for saved CSV tables.
    min_cells:
        Minimum cells per group; cell types below threshold are skipped.
    logfc_threshold:
        |log2FC| cutoff for labelling significant genes on volcano.
    padj_threshold:
        Adjusted p-value cutoff for significance.
    """

    def __init__(
        self,
        adata: AnnData,
        condition_key: str = "condition",
        disease_label: str = "disease",
        control_label: str = "control",
        cell_type_key: str = "cell_type",
        figures_dir: Path = Path("results/figures"),
        results_dir: Path = Path("results"),
        min_cells: int = 20,
        logfc_threshold: float = 1.0,
        padj_threshold: float = 0.05,
    ) -> None:
        self.adata = adata
        self.condition_key = condition_key
        self.disease_label = disease_label
        self.control_label = control_label
        self.cell_type_key = cell_type_key
        self.figures_dir = Path(figures_dir)
        self.results_dir = Path(results_dir)
        self.min_cells = min_cells
        self.logfc_threshold = logfc_threshold
        self.padj_threshold = padj_threshold

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Populated by run()
        self._deg_tables: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, pd.DataFrame]:
        """Run Wilcoxon DEG analysis for every cell type.

        Returns
        -------
        dict
            Mapping cell_type -> DataFrame with columns
            [gene, log2fc, pval, padj, wilcoxon_score].
        """
        self._validate_obs_keys()
        cell_types = self.adata.obs[self.cell_type_key].unique().tolist()
        logger.info(
            "Running DEG analysis across %d cell types.", len(cell_types)
        )

        for ct in cell_types:
            logger.info("  Cell type: %s", ct)
            subset = self._subset_cell_type(ct)
            if subset is None:
                continue
            deg_df = self._run_wilcoxon(subset, ct)
            if deg_df is not None:
                self._deg_tables[ct] = deg_df
                self._save_deg_table(deg_df, ct)

        logger.info(
            "DEG analysis complete for %d cell types.", len(self._deg_tables)
        )
        return self._deg_tables

    def plot_volcano(
        self,
        cell_type: str,
        top_n_labels: int = 20,
    ) -> Path:
        """Generate and save a volcano plot for a given cell type.

        Parameters
        ----------
        cell_type:
            Must have been analysed in ``run()``.
        top_n_labels:
            Number of top genes (by |log2fc| * -log10 padj) to label.

        Returns
        -------
        Path
            Saved figure path.
        """
        if cell_type not in self._deg_tables:
            raise KeyError(
                f"Cell type '{cell_type}' not found. Available: "
                f"{list(self._deg_tables.keys())}"
            )
        df = self._deg_tables[cell_type].copy()
        safe_name = cell_type.replace(" ", "_").replace("/", "-")
        fig_path = self.figures_dir / f"volcano_{safe_name}.png"

        df["neg_log10_padj"] = -np.log10(df[_PADJ_COL].clip(lower=1e-300))
        df["significant"] = (
            (df[_PADJ_COL] < self.padj_threshold)
            & (df[_LOG2FC_COL].abs() >= self.logfc_threshold)
        )
        df["color"] = "grey"
        df.loc[
            (df["significant"]) & (df[_LOG2FC_COL] > 0), "color"
        ] = "firebrick"
        df.loc[
            (df["significant"]) & (df[_LOG2FC_COL] < 0), "color"
        ] = "steelblue"

        # Top genes to label
        df["rank_metric"] = df[_LOG2FC_COL].abs() * df["neg_log10_padj"]
        top_genes = df.nlargest(top_n_labels, "rank_metric")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            df[_LOG2FC_COL],
            df["neg_log10_padj"],
            c=df["color"],
            s=8,
            alpha=0.6,
            linewidths=0,
        )

        # Threshold lines
        ax.axhline(
            -np.log10(self.padj_threshold), color="black", linestyle="--", lw=0.8
        )
        ax.axvline(self.logfc_threshold, color="black", linestyle="--", lw=0.8)
        ax.axvline(-self.logfc_threshold, color="black", linestyle="--", lw=0.8)

        # Gene labels
        for _, row in top_genes.iterrows():
            ax.text(
                row[_LOG2FC_COL],
                row["neg_log10_padj"],
                row[_GENE_COL],
                fontsize=6,
                ha="center",
                va="bottom",
            )

        n_up = int((df["color"] == "firebrick").sum())
        n_dn = int((df["color"] == "steelblue").sum())
        ax.set_xlabel("log2 Fold Change (disease / control)", fontsize=11)
        ax.set_ylabel("-log10 adjusted p-value", fontsize=11)
        ax.set_title(
            f"Volcano: {cell_type}  |  up={n_up}  down={n_dn}", fontsize=12
        )

        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Volcano plot saved to %s.", fig_path)
        return fig_path

    def plot_all_volcanos(self, top_n_labels: int = 20) -> List[Path]:
        """Generate volcano plots for all analysed cell types.

        Returns
        -------
        list of Path
        """
        paths = []
        for ct in self._deg_tables:
            try:
                p = self.plot_volcano(ct, top_n_labels=top_n_labels)
                paths.append(p)
            except Exception as exc:
                logger.error("Failed volcano for '%s': %s", ct, exc)
        return paths

    @property
    def deg_tables(self) -> Dict[str, pd.DataFrame]:
        """Return all DEG tables keyed by cell type."""
        return self._deg_tables

    def get_top_degs(
        self,
        cell_type: str,
        n: int = 50,
        direction: str = "both",
    ) -> pd.DataFrame:
        """Return top DEGs ranked by |log2FC|.

        Parameters
        ----------
        cell_type:
            Cell type label as used in ``run()``.
        n:
            Number of genes to return.
        direction:
            ``"up"``, ``"down"``, or ``"both"``.
        """
        df = self._deg_tables[cell_type].copy()
        df = df[df[_PADJ_COL] < self.padj_threshold]
        if direction == "up":
            df = df[df[_LOG2FC_COL] > 0]
        elif direction == "down":
            df = df[df[_LOG2FC_COL] < 0]
        return df.nlargest(n, _LOG2FC_COL) if direction != "down" else df.nsmallest(n, _LOG2FC_COL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_obs_keys(self) -> None:
        for key in [self.condition_key, self.cell_type_key]:
            if key not in self.adata.obs.columns:
                raise KeyError(
                    f"Required obs column '{key}' not found in adata.obs. "
                    f"Available: {list(self.adata.obs.columns)}"
                )
        for label in [self.disease_label, self.control_label]:
            if label not in self.adata.obs[self.condition_key].values:
                raise ValueError(
                    f"Label '{label}' not found in adata.obs['{self.condition_key}']. "
                    f"Unique values: {self.adata.obs[self.condition_key].unique().tolist()}"
                )

    def _subset_cell_type(self, cell_type: str) -> Optional[AnnData]:
        """Return AnnData subset for a single cell type with both conditions."""
        mask = self.adata.obs[self.cell_type_key] == cell_type
        sub = self.adata[mask].copy()
        cond_counts = sub.obs[self.condition_key].value_counts()
        n_dis = cond_counts.get(self.disease_label, 0)
        n_ctrl = cond_counts.get(self.control_label, 0)
        if n_dis < self.min_cells or n_ctrl < self.min_cells:
            logger.warning(
                "Skipping '%s': disease=%d, control=%d (min=%d).",
                cell_type,
                n_dis,
                n_ctrl,
                self.min_cells,
            )
            return None
        return sub

    def _run_wilcoxon(
        self, subset: AnnData, cell_type: str
    ) -> Optional[pd.DataFrame]:
        """Run scanpy rank_genes_groups (Wilcoxon) and return DEG DataFrame."""
        try:
            sc.tl.rank_genes_groups(
                subset,
                groupby=self.condition_key,
                groups=[self.disease_label],
                reference=self.control_label,
                method="wilcoxon",
                pts=True,
                use_raw=False,
            )
        except Exception as exc:
            logger.error("rank_genes_groups failed for '%s': %s", cell_type, exc)
            return None

        return self._extract_deg_table(subset)

    def _extract_deg_table(self, subset: AnnData) -> pd.DataFrame:
        """Convert scanpy rank_genes_groups result to a tidy DataFrame."""
        rgg = subset.uns["rank_genes_groups"]
        group = self.disease_label

        genes = np.array(rgg["names"][group], dtype=str)
        scores = np.array(rgg["scores"][group], dtype=float)
        pvals = np.array(rgg["pvals"][group], dtype=float)
        padjs = np.array(rgg["pvals_adj"][group], dtype=float)
        logfcs = np.array(rgg["logfoldchanges"][group], dtype=float)

        df = pd.DataFrame(
            {
                _GENE_COL: genes,
                _LOG2FC_COL: logfcs,
                _PVAL_COL: pvals,
                _PADJ_COL: padjs,
                _SCORE_COL: scores,
            }
        )
        df.sort_values(_PADJ_COL, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _save_deg_table(self, df: pd.DataFrame, cell_type: str) -> None:
        safe_name = cell_type.replace(" ", "_").replace("/", "-")
        out_path = self.results_dir / f"DEGs_{safe_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(
            "DEG table for '%s' saved to %s (%d genes).",
            cell_type,
            out_path,
            len(df),
        )
