"""
Publication-quality figure generator for heart systemic multimodal project.

Figures produced
----------------
1.  UMAP with phenotype scores overlay
2.  Heatmap: phenotype scores x cell types
3.  Venn: cardiac secretome intersect plasma EV intersect plasma proteomics
4.  Sankey/alluvial: heart ligand -> plasma -> target organ receptor
5.  Bar chart: top 20 ranked biomarkers
6.  Cross-organ heatmap: ligands x organs
7.  HBAM distribution: violin + swarm by disease group
8.  SHAP feature importance: bar chart
9.  Network graph: heart -> plasma -> organs (networkx)
10. Multi-disease comparison radar chart

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
        hbam_scores=hbam_df,
        disease_comparison=disease_df,
        secretome_genes=secretome_set,
        plasma_ev_genes=ev_set,
        plasma_prot_genes=prot_set,
        shap_values=shap_dict,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

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
        "pdf.fonttype": 42,  # TrueType fonts in PDF (required for Nature)
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
        File formats to save (default: ``["pdf", "png"]``).
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
        """Save figure in all configured formats and close it.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save.
        stem : str
            Filename stem (no extension).

        Returns
        -------
        list[Path]
            Paths of all saved files.
        """
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
        adata: Any,
        score_key: str = "fibrosis_score",
        basis: str = "X_umap",
        color_map: str = "RdYlBu_r",
        title: str = "UMAP -- Phenotype Score Overlay",
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
    # 2. Heatmap: phenotype scores x cell types
    # ------------------------------------------------------------------

    def plot_phenotype_heatmap(
        self,
        score_matrix: pd.DataFrame,
        title: str = "Phenotype Scores by Cell Type",
    ) -> list[Path]:
        """
        Heatmap of phenotype scores (rows) x cell types (columns).

        Parameters
        ----------
        score_matrix : pd.DataFrame
            Rows = phenotype scores / gene modules, columns = cell types,
            values = mean score.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        if score_matrix.empty:
            logger.warning("score_matrix is empty -- skipping heatmap")
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
    # 3. Venn: secretome intersect plasma EV intersect plasma proteomics
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
        title: str = "Overlap: Secretome / Plasma EV / Proteomics",
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
            venn3(
                [secretome_genes, plasma_ev_genes, plasma_prot_genes],
                set_labels=labels,
                ax=ax,
                set_colors=(_PALETTE[0], _PALETTE[1], _PALETTE[2]),
                alpha=0.55,
            )
            ax.set_title(title, fontsize=13, pad=12)
            fig.tight_layout()

        except ImportError:
            logger.warning("matplotlib-venn not installed -- using bar fallback for Venn")
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
            f"{labels[0]} + {labels[1]}": len(s1 & s2),
            f"{labels[0]} + {labels[2]}": len(s1 & s3),
            f"{labels[1]} + {labels[2]}": len(s2 & s3),
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
    # 4. Sankey: heart ligand -> plasma -> target organ receptor
    # ------------------------------------------------------------------

    def plot_sankey(
        self,
        lr_pairs: pd.DataFrame,
        top_n: int = 20,
        title: str = "Heart Ligand -> Plasma -> Target Organ Receptor",
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
            logger.warning("lr_pairs is empty -- skipping Sankey")
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

            sources: list[int] = []
            targets: list[int] = []
            values: list[float] = []
            link_labels: list[str] = []

            for lig, grp in top.groupby("ligand"):
                sources.append(node_idx["heart"])
                targets.append(node_idx[str(lig)])
                values.append(float(grp["score"].sum()))
                link_labels.append(f"heart -> {lig}")
            for _, row in top.iterrows():
                sources.append(node_idx[str(row["ligand"])])
                targets.append(node_idx[str(row["receptor"])])
                values.append(float(row["score"]))
                link_labels.append(f"{row['ligand']} -> {row['receptor']}")
            for _, row in top.iterrows():
                sources.append(node_idx[str(row["receptor"])])
                targets.append(node_idx[str(row["target_tissue"])])
                values.append(float(row["score"]))
                link_labels.append(f"{row['receptor']} -> {row['target_tissue']}")

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
            logger.info("plotly not available -- generating matplotlib alluvial fallback")

        fig = self._alluvial_matplotlib(top, title)
        saved.extend(self._save(fig, "sankey_lr_axes"))
        return saved

    @staticmethod
    def _alluvial_matplotlib(top: pd.DataFrame, title: str) -> plt.Figure:
        """Simple stacked bar alluvial fallback."""
        sns.set_style(_STYLE)
        if top.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No LR data", ha="center", va="center", transform=ax.transAxes)
            return fig

        ligands = top["ligand"].value_counts().head(20)
        colors = sns.color_palette("husl", len(ligands))

        fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=False)

        axes[0].barh(ligands.index.tolist(), ligands.values, color=colors)
        axes[0].set_title("Ligands\n(heart-secreted)", fontsize=10)
        axes[0].set_xlabel("Total score")
        sns.despine(ax=axes[0])

        rec_counts = (
            top.groupby("receptor")["score"].sum().sort_values(ascending=False).head(20)
        )
        colors2 = sns.color_palette("tab20", len(rec_counts))
        axes[1].barh(rec_counts.index.tolist(), rec_counts.values, color=colors2)
        axes[1].set_title("Receptors\n(target tissues)", fontsize=10)
        axes[1].set_xlabel("Total score")
        sns.despine(ax=axes[1])

        if "target_tissue" in top.columns:
            tissue_scores = top.groupby("target_tissue")["score"].sum().sort_values()
            axes[2].barh(
                tissue_scores.index.tolist(),
                tissue_scores.values,
                color=sns.color_palette("Set2", len(tissue_scores)),
            )
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
            logger.warning("ranked_biomarkers is empty -- skipping bar chart")
            return []

        self._set_style()
        top = ranked_biomarkers.head(top_n).sort_values("composite_score")
        palette = plt.cm.get_cmap("YlOrRd")(
            top.get("secretome_score", pd.Series(0.5, index=top.index)).values
        )

        fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.40)))
        bars = ax.barh(
            top["gene"],
            top["composite_score"],
            color=palette,
            edgecolor="white",
            linewidth=0.4,
        )

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

        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Secretome Score", fontsize=9)

        sns.despine(ax=ax)
        fig.tight_layout()
        return self._save(fig, "top20_biomarkers_bar")

    # ------------------------------------------------------------------
    # 6. Cross-organ heatmap: ligands x organs
    # ------------------------------------------------------------------

    def plot_cross_organ_heatmap(
        self,
        lr_pairs: pd.DataFrame,
        top_n_ligands: int = 30,
        title: str = "Cross-Organ Signaling: Ligands x Target Organs",
    ) -> list[Path]:
        """
        Heatmap showing signaling strength between cardiac ligands and
        target organs.

        Parameters
        ----------
        lr_pairs : pd.DataFrame
            Columns: ligand, target_tissue, score.
        top_n_ligands : int
            Maximum number of ligands to show (rows).
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        required = {"ligand", "target_tissue", "score"}
        if lr_pairs.empty or not required.issubset(lr_pairs.columns):
            logger.warning("lr_pairs missing required columns for cross-organ heatmap -- skipping")
            return []

        self._set_style()

        # Aggregate: sum scores per ligand x tissue
        pivot = (
            lr_pairs.groupby(["ligand", "target_tissue"])["score"]
            .sum()
            .unstack(fill_value=0.0)
        )

        # Keep top ligands by total signal
        top_ligands = pivot.sum(axis=1).nlargest(top_n_ligands).index
        pivot = pivot.loc[top_ligands]

        n_rows, n_cols = pivot.shape
        fig_h = max(5, n_rows * 0.35)
        fig_w = max(6, n_cols * 0.8)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            linewidths=0.3,
            linecolor="#DDDDDD",
            cbar_kws={"label": "Aggregate LR Score", "shrink": 0.7},
            annot=n_rows <= 20 and n_cols <= 10,
            fmt=".2f",
        )
        ax.set_title(title, fontsize=13, pad=12)
        ax.set_xlabel("Target Organ", fontsize=11)
        ax.set_ylabel("Cardiac Ligand", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        fig.tight_layout()
        return self._save(fig, "cross_organ_heatmap")

    # ------------------------------------------------------------------
    # 7. HBAM distribution: violin + swarm by disease group
    # ------------------------------------------------------------------

    def plot_hbam_distribution(
        self,
        hbam_scores: pd.DataFrame,
        group_col: str = "disease_group",
        score_col: str = "hbam_score",
        title: str = "HBAM Index Distribution by Disease Group",
    ) -> list[Path]:
        """
        Violin + swarm overlay of HBAM scores by disease group.

        Parameters
        ----------
        hbam_scores : pd.DataFrame
            Must contain ``score_col`` and ``group_col`` columns.
        group_col : str
            Column defining disease groups (x-axis).
        score_col : str
            Column with HBAM score values (y-axis).
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        if hbam_scores.empty or score_col not in hbam_scores.columns:
            logger.warning("hbam_scores missing required column '%s' -- skipping", score_col)
            return []

        self._set_style()

        plot_df = hbam_scores[[score_col]].copy()
        has_groups = group_col in hbam_scores.columns
        if has_groups:
            plot_df[group_col] = hbam_scores[group_col].values
        else:
            plot_df[group_col] = "all"

        groups = sorted(plot_df[group_col].unique().tolist())
        palette = sns.color_palette("Set2", len(groups))

        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 6))

        sns.violinplot(
            data=plot_df,
            x=group_col,
            y=score_col,
            palette=palette,
            inner=None,
            ax=ax,
            order=groups,
            linewidth=0.8,
        )

        # Swarm only if dataset is not too large
        n_pts = len(plot_df)
        if n_pts <= 500:
            sns.swarmplot(
                data=plot_df,
                x=group_col,
                y=score_col,
                color="black",
                size=2.5,
                alpha=0.6,
                ax=ax,
                order=groups,
            )
        else:
            sns.stripplot(
                data=plot_df,
                x=group_col,
                y=score_col,
                color="black",
                size=1.5,
                alpha=0.3,
                ax=ax,
                order=groups,
                jitter=True,
            )

        ax.axhline(0.8, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="High-priority threshold (0.80)")
        ax.set_xlabel("Disease Group", fontsize=11)
        ax.set_ylabel("HBAM Score", fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        return self._save(fig, "hbam_distribution")

    # ------------------------------------------------------------------
    # 8. SHAP feature importance: bar chart
    # ------------------------------------------------------------------

    def plot_shap_importance(
        self,
        shap_values: dict[str, float],
        title: str = "HBAM Model -- SHAP Feature Importance",
    ) -> list[Path]:
        """
        Horizontal bar chart of mean absolute SHAP values per feature.

        Parameters
        ----------
        shap_values : dict[str, float]
            Mapping of feature name to mean absolute SHAP value.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        if not shap_values:
            logger.warning("shap_values is empty -- skipping SHAP importance plot")
            return []

        self._set_style()

        # Sort descending
        sorted_items = sorted(shap_values.items(), key=lambda x: x[1])
        features = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]

        palette = sns.color_palette("flare", len(features))
        fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.45)))
        bars = ax.barh(features, values, color=palette)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
                color="#333333",
            )

        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        sns.despine(ax=ax)
        fig.tight_layout()
        return self._save(fig, "shap_feature_importance")

    # ------------------------------------------------------------------
    # 9. Network graph: heart -> plasma -> organs
    # ------------------------------------------------------------------

    def plot_network_graph(
        self,
        lr_pairs: pd.DataFrame,
        top_n: int = 15,
        title: str = "Heart-to-Organ Signaling Network",
    ) -> list[Path]:
        """
        Network graph: heart -> ligand -> receptor -> target organ.

        Uses networkx for layout.  Falls back to a simple alluvial bar
        chart if networkx is not installed.

        Parameters
        ----------
        lr_pairs : pd.DataFrame
            Columns: ligand, receptor, target_tissue, score.
        top_n : int
            Number of top LR pairs to visualise.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        if lr_pairs.empty:
            logger.warning("lr_pairs is empty -- skipping network graph")
            return []

        cols = [c for c in ["ligand", "receptor", "target_tissue", "score"] if c in lr_pairs.columns]
        top = lr_pairs[cols].head(top_n).copy()

        try:
            import networkx as nx  # noqa: PLC0415
        except ImportError:
            logger.warning("networkx not installed -- skipping network graph")
            return []

        self._set_style()

        G = nx.DiGraph()
        G.add_node("heart", layer=0, node_type="source")

        for _, row in top.iterrows():
            lig = str(row["ligand"])
            rec = str(row["receptor"])
            tissue = str(row["target_tissue"])
            score = float(row.get("score", 1.0))

            G.add_node(lig, layer=1, node_type="ligand")
            G.add_node(rec, layer=2, node_type="receptor")
            G.add_node(tissue, layer=3, node_type="organ")

            G.add_edge("heart", lig, weight=score)
            G.add_edge(lig, rec, weight=score)
            G.add_edge(rec, tissue, weight=score)

        # Multipartite layout
        pos = nx.multipartite_layout(G, subset_key="layer", scale=2.0)

        node_colors = {
            "source": "#E63946",
            "ligand": "#457B9D",
            "receptor": "#2A9D8F",
            "organ": "#E9C46A",
        }
        colors = [node_colors.get(G.nodes[n].get("node_type", "ligand"), "#AAAAAA") for n in G.nodes]
        edge_weights = [G[u][v]["weight"] for u, v in G.edges]
        max_w = max(edge_weights) if edge_weights else 1.0
        edge_widths = [1.0 + 3.0 * (w / max_w) for w in edge_weights]

        fig, ax = plt.subplots(figsize=(14, 8))
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.5,
            edge_color="#555555",
            arrows=True,
            arrowsize=12,
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

        # Legend
        legend_handles = [
            mpatches.Patch(color=c, label=lbl.capitalize())
            for lbl, c in node_colors.items()
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=False)
        ax.set_title(title, fontsize=13, pad=12)
        ax.axis("off")
        fig.tight_layout()
        return self._save(fig, "network_graph_heart_organs")

    # ------------------------------------------------------------------
    # 10. Multi-disease comparison radar chart
    # ------------------------------------------------------------------

    def plot_disease_radar(
        self,
        disease_comparison: pd.DataFrame,
        score_cols: Optional[list[str]] = None,
        title: str = "Multi-Disease Comparison -- Secretome Score Radar",
    ) -> list[Path]:
        """
        Radar (spider) chart comparing secretome/HBAM scores across diseases.

        Parameters
        ----------
        disease_comparison : pd.DataFrame
            Must contain a ``disease`` column and numeric score columns.
        score_cols : list[str], optional
            Columns to use as radar axes.  Defaults to all numeric columns
            except ``disease``.
        title : str
            Figure title.

        Returns
        -------
        list[Path]
        """
        if disease_comparison.empty or "disease" not in disease_comparison.columns:
            logger.warning("disease_comparison missing 'disease' column -- skipping radar chart")
            return []

        self._set_style()

        numeric_cols = disease_comparison.select_dtypes(include="number").columns.tolist()
        axes_cols = score_cols if score_cols else numeric_cols
        axes_cols = [c for c in axes_cols if c in disease_comparison.columns]

        if not axes_cols:
            logger.warning("No numeric columns for radar chart -- skipping")
            return []

        # Mean score per disease
        grouped = disease_comparison.groupby("disease")[axes_cols].mean()
        diseases = grouped.index.tolist()
        n_axes = len(axes_cols)

        if n_axes < 3:
            logger.warning("Need at least 3 axes for radar chart, got %d -- skipping", n_axes)
            return []

        angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        palette = sns.color_palette("tab10", len(diseases))

        for disease, color in zip(diseases, palette):
            values = grouped.loc[disease, axes_cols].tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=1.5, color=color, label=disease)
            ax.fill(angles, values, alpha=0.10, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [c.replace("_", " ").title() for c in axes_cols],
            fontsize=8,
        )
        ax.set_title(title, fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8, frameon=False)
        fig.tight_layout()
        return self._save(fig, "disease_radar_chart")

    # ------------------------------------------------------------------
    # Run all figures
    # ------------------------------------------------------------------

    def run_all(
        self,
        adata: Any = None,
        ranked_biomarkers: Optional[pd.DataFrame] = None,
        lr_pairs: Optional[pd.DataFrame] = None,
        hbam_scores: Optional[pd.DataFrame] = None,
        disease_comparison: Optional[pd.DataFrame] = None,
        shap_values: Optional[dict[str, float]] = None,
        secretome_genes: Optional[set[str]] = None,
        plasma_ev_genes: Optional[set[str]] = None,
        plasma_prot_genes: Optional[set[str]] = None,
        score_matrix: Optional[pd.DataFrame] = None,
        score_key: str = "fibrosis_score",
        top_n: int = 20,
    ) -> dict[str, list[Path]]:
        """
        Generate all figures and return a dict of label to file paths.

        Parameters
        ----------
        adata : AnnData, optional
            For UMAP figures.
        ranked_biomarkers : pd.DataFrame, optional
        lr_pairs : pd.DataFrame, optional
        hbam_scores : pd.DataFrame, optional
            Columns: gene, hbam_score, hbam_percentile, disease_group.
        disease_comparison : pd.DataFrame, optional
            Columns: disease, plus numeric score columns.
        shap_values : dict[str, float], optional
            Feature name -> mean absolute SHAP value.
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
            Keys: umap, heatmap, venn, sankey, bar, cross_organ_heatmap,
                  hbam_distribution, shap_importance, network_graph,
                  disease_radar.
        """
        paths: dict[str, list[Path]] = {}

        if adata is not None:
            logger.info("Generating UMAP figure...")
            p = self.plot_umap_phenotype(adata, score_key=score_key)
            if p:
                paths["umap"] = p

        if score_matrix is not None and not score_matrix.empty:
            logger.info("Generating phenotype heatmap...")
            p = self.plot_phenotype_heatmap(score_matrix)
            if p:
                paths["heatmap"] = p

        if any(g is not None for g in [secretome_genes, plasma_ev_genes, plasma_prot_genes]):
            logger.info("Generating Venn diagram...")
            p = self.plot_venn(
                secretome_genes or set(),
                plasma_ev_genes or set(),
                plasma_prot_genes or set(),
            )
            if p:
                paths["venn"] = p

        if lr_pairs is not None and not lr_pairs.empty:
            logger.info("Generating Sankey diagram...")
            p = self.plot_sankey(lr_pairs, top_n=top_n)
            if p:
                paths["sankey"] = p

            logger.info("Generating cross-organ heatmap...")
            p = self.plot_cross_organ_heatmap(lr_pairs)
            if p:
                paths["cross_organ_heatmap"] = p

            logger.info("Generating network graph...")
            p = self.plot_network_graph(lr_pairs, top_n=top_n)
            if p:
                paths["network_graph"] = p

        if ranked_biomarkers is not None and not ranked_biomarkers.empty:
            logger.info("Generating top biomarkers bar chart...")
            p = self.plot_top_biomarkers_bar(ranked_biomarkers, top_n=top_n)
            if p:
                paths["bar"] = p

        if hbam_scores is not None and not hbam_scores.empty:
            logger.info("Generating HBAM distribution plot...")
            p = self.plot_hbam_distribution(hbam_scores)
            if p:
                paths["hbam_distribution"] = p

        if shap_values:
            logger.info("Generating SHAP feature importance plot...")
            p = self.plot_shap_importance(shap_values)
            if p:
                paths["shap_importance"] = p

        if disease_comparison is not None and not disease_comparison.empty:
            logger.info("Generating multi-disease radar chart...")
            p = self.plot_disease_radar(disease_comparison)
            if p:
                paths["disease_radar"] = p

        total = sum(len(v) for v in paths.values())
        logger.info(
            "FigureGenerator: %d figure files saved across %d panels",
            total,
            len(paths),
        )
        return paths
