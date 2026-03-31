"""
Publication-quality figure generator for heart systemic multimodal project.

Figures produced
----------------
1. UMAP with phenotype scores overlay
2. Heatmap: phenotype scores × cell types
3. Venn: cardiac secretome ∩ plasma EV ∩ plasma proteomics
4. Sankey/alluvial: heart ligand → plasma → target organ receptor
5. Bar chart: top 20 ranked biomarkers

All figures:
- 300 DPI, PDF + PNG
- Consistent publication style (seaborn whitegrid, Nature-compatible palette)

Usage
-----
    fig_gen = FigureGenerator(figures_dir=Path("results/figures"))
    paths = fig_gen.run_all(
        adata=adata,
        ranked_biomarkers=ranked_df,
        lr_pairs=lr_df,
        secretome_genes=secretome_set,
        plasma_ev_genes=ev_set,
        plasma_prot_genes=prot_set,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,   # TrueType fonts in PDF (required for Nature)
        "ps.fonttype": 42,
    }
)

logger = logging.getLogger(__name__)

_DPI = 300
_FORMATS = ["pdf", "png"]
_PALETTE = sns.color_palette("tab10")
_STYLE = "whitegrid"


class FigureGenerator:
    """
    Generate all publication figures for the heart multimodal project.

    Parameters
    ----------
    figures_dir : Path
        Output directory for all figures.
    formats : list[str]
        File formats to save (default: ["pdf", "png"]).
    dpi : int
        Resolution for raster outputs (default: 300).
    """

    def __init__(
        self,
        figures_dir: Path = Path("results/figures"),
        formats: Optional[list[str]] = None,
        dpi: int = _DPI,
    ) -> None:
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats if formats is not None else list(_FORMATS)
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, stem: str) -> list[Path]:
        """Save figure in all configured formats."""
        saved: list[Path] = []
        for fmt in self.formats:
            out = self.figures_dir / f"{stem}.{fmt}"
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
            saved.append(out)
            logger.debug("Saved %s", out)
        plt.close(fig)
        return saved

    @staticmethod
    def _set_style() -> None:
        sns.set_style(_STYLE)
        sns.set_context("paper")

    # ------------------------------------------------------------------
    # 1. UMAP with phenotype scores overlay
    # ------------------------------------------------------------------

    def plot_umap_phenotype(
        self,
        adata,
        score_key: str = "fibrosis_score",
        basis: str = "X_umap",
        color_map: str = "RdYlBu_r",
        title: str = "UMAP — Phenotype Score Overlay",
    ) -> list[Path]:
        """
        Plot UMAP embedding coloured by a continuous phenotype score.

        Parameters
        ----------
        adata : AnnData
            Must have ``adata.obsm[basis]`` (UMAP coordinates) and
            ``adata.obs[score_key]``.
        score_key : str
            Column in ``adata.obs`` with the score to colour by.
        basis : str
            Key in ``adata.obsm`` for 2-D embedding.
        color_map : str
            Matplotlib colormap.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        self._set_style()
        try:
            coords = adata.obsm[basis]
            scores = adata.obs[score_key].values
        except (AttributeError, KeyError) as exc:
            logger.error("Cannot plot UMAP: %s", exc)
            return []

        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=scores,
            cmap=color_map,
            s=3,
            alpha=0.7,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(score_key.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
        fig.tight_layout()
        return self._save(fig, "umap_phenotype_scores")

    # ------------------------------------------------------------------
    # 2. Heatmap: phenotype scores × cell types
    # ------------------------------------------------------------------

    def plot_phenotype_heatmap(
        self,
        score_matrix: pd.DataFrame,
        title: str = "Phenotype Scores by Cell Type",
    ) -> list[Path]:
        """
        Heatmap of phenotype scores (rows) × cell types (columns).

        Parameters
        ----------
        score_matrix : pd.DataFrame
            Rows = phenotype scores / gene modules, columns = cell types,
            values = mean score.

        Returns
        -------
        list[Path]
        """
        if score_matrix.empty:
            logger.warning("score_matrix is empty — skipping heatmap")
            return []

        self._set_style()
        n_rows, n_cols = score_matrix.shape
        fig_h = max(4, n_rows * 0.45)
        fig_w = max(6, n_cols * 0.55)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            score_matrix,
            ax=ax,
            cmap="RdYlBu_r",
            center=0,
            annot=n_rows <= 20 and n_cols <= 20,
            fmt=".2f",
            linewidths=0.3,
            linecolor="#DDDDDD",
            cbar_kws={"label": "Score", "shrink": 0.7},
        )
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("Cell Type", fontsize=11)
        ax.set_ylabel("Phenotype / Module", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        fig.tight_layout()
        return self._save(fig, "phenotype_heatmap")

    # ------------------------------------------------------------------
    # 3. Venn: secretome ∩ plasma EV ∩ plasma proteomics
    # ------------------------------------------------------------------

    def plot_venn(
        self,
        secretome_genes: set[str],
        plasma_ev_genes: set[str],
        plasma_prot_genes: set[str],
        labels: tuple[str, str, str] = (
            "Cardiac Secretome",
            "Plasma EV",
            "Plasma Proteomics",
        ),
        title: str = "Overlap: Secretome ∩ Plasma EV ∩ Proteomics",
    ) -> list[Path]:
        """
        Three-way Venn diagram of gene set overlaps.

        Falls back to a proportional bar chart if matplotlib-venn is not
        installed.

        Parameters
        ----------
        secretome_genes, plasma_ev_genes, plasma_prot_genes : set[str]
            Gene symbol sets for each layer.
        labels : tuple[str, str, str]
            Circle labels.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        self._set_style()
        try:
            from matplotlib_venn import venn3  # noqa: PLC0415

            fig, ax = plt.subplots(figsize=(7, 6))
            v = venn3(
                [secretome_genes, plasma_ev_genes, plasma_prot_genes],
                set_labels=labels,
                ax=ax,
                set_colors=(_PALETTE[0], _PALETTE[1], _PALETTE[2]),
                alpha=0.55,
            )
            ax.set_title(title, fontsize=13, pad=12)
            fig.tight_layout()

        except ImportError:
            logger.warning("matplotlib-venn not installed — using bar fallback for Venn")
            fig = self._venn_bar_fallback(
                secretome_genes, plasma_ev_genes, plasma_prot_genes, labels, title
            )

        return self._save(fig, "venn_secretome_plasma")

    @staticmethod
    def _venn_bar_fallback(
        s1: set[str],
        s2: set[str],
        s3: set[str],
        labels: tuple[str, str, str],
        title: str,
    ) -> plt.Figure:
        """Bar chart fallback when matplotlib-venn is unavailable."""
        sizes = {
            labels[0]: len(s1),
            labels[1]: len(s2),
            labels[2]: len(s3),
            f"{labels[0]} ∩ {labels[1]}": len(s1 & s2),
            f"{labels[0]} ∩ {labels[2]}": len(s1 & s3),
            f"{labels[1]} ∩ {labels[2]}": len(s2 & s3),
            "All three": len(s1 & s2 & s3),
        }
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(list(sizes.keys()), list(sizes.values()), color=_PALETTE[:7])
        ax.set_xlabel("Number of Genes", fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. Sankey: heart ligand → plasma → target organ receptor
    # ------------------------------------------------------------------

    def plot_sankey(
        self,
        lr_pairs: pd.DataFrame,
        top_n: int = 20,
        title: str = "Heart Ligand → Plasma → Target Organ Receptor",
    ) -> list[Path]:
        """
        Sankey / alluvial diagram of top LR axes.

        Uses plotly for interactive HTML + a static matplotlib fallback.

        Parameters
        ----------
        lr_pairs : pd.DataFrame
            Columns: ligand, receptor, target_tissue, score.
        top_n : int
            Number of top LR axes to display.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
            Saved figure paths (PNG + PDF from matplotlib; HTML from plotly
            if available).
        """
        if lr_pairs.empty:
            logger.warning("lr_pairs is empty — skipping Sankey")
            return []

        cols = [c for c in ["ligand", "receptor", "target_tissue", "score"] if c in lr_pairs.columns]
        top = lr_pairs[cols].head(top_n).copy()

        saved: list[Path] = []

        # --- Try plotly Sankey ---
        try:
            import plotly.graph_objects as go  # noqa: PLC0415

            nodes_raw = (
                ["heart"]
                + top["ligand"].unique().tolist()
                + top["receptor"].unique().tolist()
                + top["target_tissue"].unique().tolist()
            )
            node_list: list[str] = list(dict.fromkeys(nodes_raw))
            node_idx = {n: i for i, n in enumerate(node_list)}

            sources, targets, values, link_labels = [], [], [], []
            # heart → ligand
            for lig, grp in top.groupby("ligand"):
                sources.append(node_idx["heart"])
                targets.append(node_idx[lig])
                values.append(float(grp["score"].sum()))
                link_labels.append(f"heart → {lig}")
            # ligand → receptor
            for _, row in top.iterrows():
                sources.append(node_idx[row["ligand"]])
                targets.append(node_idx[row["receptor"]])
                values.append(float(row["score"]))
                link_labels.append(f"{row['ligand']} → {row['receptor']}")
            # receptor → tissue
            for _, row in top.iterrows():
                sources.append(node_idx[row["receptor"]])
                targets.append(node_idx[row["target_tissue"]])
                values.append(float(row["score"]))
                link_labels.append(f"{row['receptor']} → {row['target_tissue']}")

            fig_pl = go.Figure(
                go.Sankey(
                    node=dict(label=node_list, pad=15, thickness=15),
                    link=dict(source=sources, target=targets, value=values, label=link_labels),
                )
            )
            fig_pl.update_layout(title_text=title, font_size=10)
            html_path = self.figures_dir / "sankey_lr_axes.html"
            fig_pl.write_html(str(html_path))
            saved.append(html_path)
            logger.info("Saved interactive Sankey to %s", html_path)
        except ImportError:
            logger.info("plotly not available — generating matplotlib alluvial fallback")

        # --- Matplotlib alluvial fallback ---
        fig = self._alluvial_matplotlib(top, title)
        saved.extend(self._save(fig, "sankey_lr_axes"))
        return saved

    @staticmethod
    def _alluvial_matplotlib(top: pd.DataFrame, title: str) -> plt.Figure:
        """Simple stacked bar 'alluvial' fallback."""
        sns.set_style(_STYLE)
        if top.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No LR data", ha="center", va="center", transform=ax.transAxes)
            return fig

        ligands = top["ligand"].value_counts().head(20)
        colors = sns.color_palette("husl", len(ligands))

        fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=False)

        # Column 1: Ligand totals
        axes[0].barh(ligands.index.tolist(), ligands.values, color=colors)
        axes[0].set_title("Ligands\n(heart-secreted)", fontsize=10)
        axes[0].set_xlabel("Total score")
        sns.despine(ax=axes[0])

        # Column 2: Receptor counts per ligand
        rec_counts = (
            top.groupby("receptor")["score"].sum().sort_values(ascending=False).head(20)
        )
        colors2 = sns.color_palette("tab20", len(rec_counts))
        axes[1].barh(rec_counts.index.tolist(), rec_counts.values, color=colors2)
        axes[1].set_title("Receptors\n(target tissues)", fontsize=10)
        axes[1].set_xlabel("Total score")
        sns.despine(ax=axes[1])

        # Column 3: Tissue breakdown
        if "target_tissue" in top.columns:
            tissue_scores = top.groupby("target_tissue")["score"].sum().sort_values()
            axes[2].barh(tissue_scores.index.tolist(), tissue_scores.values,
                         color=sns.color_palette("Set2", len(tissue_scores)))
            axes[2].set_title("Target Tissues", fontsize=10)
            axes[2].set_xlabel("Total score")
            sns.despine(ax=axes[2])

        fig.suptitle(title, fontsize=13, y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 5. Bar chart: top 20 ranked biomarkers
    # ------------------------------------------------------------------

    def plot_top_biomarkers_bar(
        self,
        ranked_biomarkers: pd.DataFrame,
        top_n: int = 20,
        title: str = "Top Ranked Cardiac Biomarker Candidates",
    ) -> list[Path]:
        """
        Horizontal bar chart of top N ranked biomarkers.

        Bars are coloured by secretome_score.

        Parameters
        ----------
        ranked_biomarkers : pd.DataFrame
            Output of BiomarkerRanker.rank().
        top_n : int
            Number of top candidates to show.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        if ranked_biomarkers.empty:
            logger.warning("ranked_biomarkers is empty — skipping bar chart")
            return []

        self._set_style()
        top = ranked_biomarkers.head(top_n).sort_values("composite_score")
        palette = sns.color_palette(
            "YlOrRd",
            as_cmap=True,
        )(top.get("secretome_score", pd.Series(0.5, index=top.index)).values)

        fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.40)))
        bars = ax.barh(
            top["gene"],
            top["composite_score"],
            color=palette,
            edgecolor="white",
            linewidth=0.4,
        )

        # Annotate rank
        for bar, rank in zip(bars, top["rank"].tolist()):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"#{rank}",
                va="center",
                fontsize=7,
                color="#555555",
            )

        ax.set_xlabel("Composite Ranking Score", fontsize=11)
        ax.set_ylabel("Gene Symbol", fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlim(0, 1.1)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

        # Colorbar legend
        sm = plt.cm.ScalarMappable(
            cmap="YlOrRd",
            norm=plt.Normalize(0, 1),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Secretome Score", fontsize=9)

        sns.despine(ax=ax)
        fig.tight_layout()
        return self._save(fig, "top20_biomarkers_bar")

    # ------------------------------------------------------------------
    # Run all figures
    # ------------------------------------------------------------------

    def run_all(
        self,
        adata=None,
        ranked_biomarkers: Optional[pd.DataFrame] = None,
        lr_pairs: Optional[pd.DataFrame] = None,
        secretome_genes: Optional[set[str]] = None,
        plasma_ev_genes: Optional[set[str]] = None,
        plasma_prot_genes: Optional[set[str]] = None,
        score_matrix: Optional[pd.DataFrame] = None,
        score_key: str = "fibrosis_score",
        top_n: int = 20,
    ) -> dict[str, list[Path]]:
        """
        Generate all figures and return a dict of label → file paths.

        Parameters
        ----------
        adata : AnnData, optional
            For UMAP figures.
        ranked_biomarkers : pd.DataFrame, optional
        lr_pairs : pd.DataFrame, optional
        secretome_genes, plasma_ev_genes, plasma_prot_genes : set[str], optional
            For Venn diagram.
        score_matrix : pd.DataFrame, optional
            For heatmap (rows=modules, cols=cell types).
        score_key : str
            Column in adata.obs for UMAP colouring.
        top_n : int
            Top N for bar and Sankey plots.

        Returns
        -------
        dict[str, list[Path]]
            Keys: umap, heatmap, venn, sankey, bar.
        """
        paths: dict[str, list[Path]] = {}

        if adata is not None:
            logger.info("Generating UMAP figure…")
            p = self.plot_umap_phenotype(adata, score_key=score_key)
            if p:
                paths["umap"] = p

        if score_matrix is not None and not score_matrix.empty:
            logger.info("Generating phenotype heatmap…")
            p = self.plot_phenotype_heatmap(score_matrix)
            if p:
                paths["heatmap"] = p

        if any(g is not None for g in [secretome_genes, plasma_ev_genes, plasma_prot_genes]):
            logger.info("Generating Venn diagram…")
            p = self.plot_venn(
                secretome_genes or set(),
                plasma_ev_genes or set(),
                plasma_prot_genes or set(),
            )
            if p:
                paths["venn"] = p

        if lr_pairs is not None and not lr_pairs.empty:
            logger.info("Generating Sankey diagram…")
            p = self.plot_sankey(lr_pairs, top_n=top_n)
            if p:
                paths["sankey"] = p

        if ranked_biomarkers is not None and not ranked_biomarkers.empty:
            logger.info("Generating top biomarkers bar chart…")
            p = self.plot_top_biomarkers_bar(ranked_biomarkers, top_n=top_n)
            if p:
                paths["bar"] = p

        total = sum(len(v) for v in paths.values())
        logger.info("FigureGenerator: %d figure files saved across %d panels", total, len(paths))
        return paths
