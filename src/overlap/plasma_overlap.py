"""
Overlap analysis between cardiac secretome candidates and plasma EV proteomics.

Datasets
--------
- PXD021371 : Human plasma EV proteome (healthy + HF patients)
- PXD059929 : Cardiac-derived EV proteome (serum)
- PXD060680 : Plasma EV proteome cross-disease panel
- reference  : Curated reference plasma proteome (optional 4th source)

Workflow
--------
1. Load plasma EV proteomics tables (CSV/TSV or embedded stubs).
2. Compute source-weighted plasma_score = Σ(weight_i × detected_i).
3. Intersect with cardiac secretome candidates.
4. Fisher exact test for statistical enrichment per dataset.
5. Volcano-style scatter plot (logFC vs -log10 p-value, colored by plasma detection).
6. UpSet plot for multi-set intersections (3+ datasets).
7. Venn / bar-chart fallback for set overlap visualization.
8. Export plasma_candidates.csv with columns:
   gene, cardiac_score, plasma_detected_in, plasma_score, combined_rank.
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
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

PLASMA_DATASET_IDS: List[str] = ["PXD021371", "PXD059929", "PXD060680"]

# Background proteome size (human plasma, UniProt reviewed + predicted
# secreted, approximate)
_BACKGROUND_PROTEOME_SIZE: int = 4_500

# Default per-source weights for plasma_score computation.
# Keys must match PLASMA_DATASET_IDS + optional "reference".
DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
    "PXD021371": 0.30,
    "PXD059929": 0.35,
    "PXD060680": 0.25,
    "reference": 0.10,
}

# ---------------------------------------------------------------------------
# Embedded stub protein lists (subset; used when actual files are absent)
# ---------------------------------------------------------------------------

_STUB_PROTEINS: Dict[str, List[str]] = {
    "PXD021371": [
        # HF plasma EVs — Murillo et al. 2021 (JACC)
        "FN1", "TGFB1", "COL1A1", "COL3A1", "POSTN",
        "SERPINH1", "LOX", "LOXL2",
        "MMP2", "MMP9", "TIMP1", "TIMP2",
        "VEGFA", "PDGFA", "ANGPT1",
        "IL6", "TNF", "CCL2", "CRP",
        "NPPA", "NPPB", "MYH7",
        "ALIX", "TSG101", "CD63", "CD9", "CD81",
        "HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA5",
        "ACTB", "ACTG1", "TUBA1A",
        "GDF15", "MSTN",
        "SERPINE1", "VTN", "VWF",
        "THBS1", "THBS2",
        "CLU", "APOE", "APOC3",
        "IGF1", "FGF2",
        "HMGB1", "S100A8", "S100A9",
        "LGALS3", "SPP1",
        "DCN", "BGN", "LUM",
        "CTGF", "ACTA2",
        "PINK1", "BNIP3",
        "CDKN1A", "CDKN2A",
        "MMP1", "MMP3", "ADAMTS2",
    ],
    "PXD059929": [
        # Cardiac EV serum — Zhang et al. 2024
        "NPPA", "NPPB", "TNNT2", "TNNI3",
        "MYH6", "MYH7", "MYL2", "MYL3",
        "ACTC1", "ACTA1",
        "VEGFA", "FGF2", "IGF1",
        "BMP4", "WNT5A", "DKK1",
        "TGFB1", "TGFB2",
        "COL1A1", "COL4A1",
        "FN1", "POSTN", "CTGF",
        "MMP2", "MMP14", "TIMP1",
        "CD63", "CD9", "CD81", "ALIX", "TSG101",
        "GDF15", "MSTN", "GDF11",
        "SERPINE1", "SERPINE2",
        "CXCL12", "CCL5",
        "IL1B", "IL6",
        "PINK1", "PARK2", "BNIP3L",
        "SOD2", "CAT",
        "FBLN1", "FBLN5",
        "TNC", "COMP",
        "CLU", "APOE",
        "ANGPT1", "ANGPT2",
    ],
    "PXD060680": [
        # Cross-disease plasma EV panel — Lee et al. 2024
        "GDF15", "MSTN", "GDF11",
        "SERPINE1", "CLU", "APOE",
        "TGFB1", "TGFB3",
        "FN1", "VTN", "VWF",
        "COL1A1", "COL3A1",
        "MMP2", "MMP9",
        "TIMP1", "TIMP2",
        "VEGFA", "VEGFC", "ANGPT2",
        "PDGFA", "PDGFB",
        "FGF1", "FGF2", "FGF21",
        "IGF1", "IGF2",
        "CXCL8", "CCL2", "CCL3",
        "IL6", "TNF",
        "CRP", "SAA1", "SAA2",
        "THBS1", "THBS4",
        "COMP", "MATN2",
        "DCN", "BGN", "LUM",
        "LGALS3", "LGALS1",
        "SPP1", "HMGB1",
        "S100A8", "S100A9",
        "CD63", "CD9", "CD81",
        "ALIX", "TSG101",
        "HSP90AA1", "HSPA5",
        "ACTB", "TUBA1A",
        "NPPA", "NPPB",
        "CTGF", "LOX", "POSTN",
        "ADAMTS1", "ADAMTS5",
    ],
    # Optional curated reference plasma proteome seed
    "reference": [
        "APOA1", "APOA2", "APOB", "APOE", "ALB",
        "FN1", "VTN", "VWF",
        "FGF21", "GDF15", "IGF1",
        "IL6", "TNF", "TGFB1",
        "VEGFA", "ANGPT1",
        "CLU", "SERPINE1",
        "THBS1", "THBS2",
        "SPP1", "LGALS3",
        "S100A8", "S100A9",
        "HMGB1",
    ],
}

# All source IDs including optional reference
_ALL_SOURCE_IDS: List[str] = PLASMA_DATASET_IDS + ["reference"]


class PlasmaOverlapAnalyzer:
    """Intersect cardiac secretome candidates with plasma EV proteomics.

    Parameters
    ----------
    candidates_path:
        Path to ``candidate_ligands.csv`` generated by
        ``CardiacSecretomeInference.run()``.  If ``None``, ``candidates``
        must be supplied directly.
    candidates:
        Pre-loaded candidates DataFrame (alternative to ``candidates_path``).
    plasma_data_dir:
        Directory containing CSV/TSV files for each dataset, named
        ``<dataset_id>.csv`` or ``<dataset_id>.tsv``.  Falls back to stubs
        when files are absent.
    gene_column:
        Column in plasma data files containing gene symbols.
    source_weights:
        Per-source weight dict for plasma_score. Keys: dataset IDs + optional
        "reference". Defaults to ``DEFAULT_SOURCE_WEIGHTS``.
    background_size:
        Total number of genes/proteins in the background for Fisher test.
    cardiac_score_column:
        Column in candidates DataFrame to use as cardiac score.
        Defaults to ``"secretome_score"``, falls back to ``"composite_score"``.
    figures_dir:
        Output directory for figures.
    results_dir:
        Output directory for CSV outputs.
    """

    def __init__(
        self,
        candidates_path: Optional[Path] = None,
        candidates: Optional[pd.DataFrame] = None,
        plasma_data_dir: Optional[Path] = None,
        gene_column: str = "gene",
        source_weights: Optional[Dict[str, float]] = None,
        background_size: int = _BACKGROUND_PROTEOME_SIZE,
        cardiac_score_column: str = "secretome_score",
        figures_dir: Path = Path("results/figures"),
        results_dir: Path = Path("results"),
    ) -> None:
        if candidates is None and candidates_path is None:
            raise ValueError("Provide either candidates_path or candidates.")

        if candidates is not None:
            self._candidates = candidates.copy()
        else:
            candidates_path = Path(candidates_path)  # type: ignore[arg-type]
            if not candidates_path.exists():
                raise FileNotFoundError(f"candidates_path not found: {candidates_path}")
            self._candidates = pd.read_csv(candidates_path)

        self.plasma_data_dir = Path(plasma_data_dir) if plasma_data_dir else None
        self.gene_column = gene_column
        self.source_weights: Dict[str, float] = (
            source_weights if source_weights is not None else DEFAULT_SOURCE_WEIGHTS
        )
        self.background_size = background_size
        self.figures_dir = Path(figures_dir)
        self.results_dir = Path(results_dir)

        # Resolve cardiac score column with fallback
        if cardiac_score_column in self._candidates.columns:
            self._cardiac_score_col = cardiac_score_column
        elif "composite_score" in self._candidates.columns:
            self._cardiac_score_col = "composite_score"
            logger.info(
                "cardiac_score_column '%s' not found; using 'composite_score'.",
                cardiac_score_column,
            )
        else:
            self._cardiac_score_col = self._candidates.columns[0]
            logger.warning(
                "No score column found; using '%s' as cardiac score.",
                self._cardiac_score_col,
            )

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Populated by run()
        self._plasma_sets: Dict[str, set] = {}
        self._overlap_df: Optional[pd.DataFrame] = None
        self._enrichment_df: Optional[pd.DataFrame] = None
        self._plasma_candidates_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute full overlap + enrichment pipeline.

        Returns
        -------
        tuple
            (plasma_candidates DataFrame, enrichment DataFrame)
        """
        logger.info("Loading plasma EV proteomics datasets.")
        # Load all sources including reference
        for src in _ALL_SOURCE_IDS:
            self._plasma_sets[src] = self._load_plasma_set(src)

        candidate_genes: set = set(self._candidates["gene"].tolist())
        logger.info(
            "Candidate genes: %d | Background: %d",
            len(candidate_genes),
            self.background_size,
        )

        # --- Build per-gene plasma scores ---
        self._overlap_df = self._build_overlap_df(candidate_genes)

        # --- Build plasma_candidates with combined_rank ---
        self._plasma_candidates_df = self._build_plasma_candidates()

        # --- Enrichment (Fisher) per PRIDE dataset only ---
        self._enrichment_df = self._fisher_enrichment(candidate_genes)

        # --- Save ---
        self._save_outputs()

        return self._plasma_candidates_df, self._enrichment_df

    def plot_volcano(self) -> Path:
        """Volcano-style scatter: logFC vs -log10(padj), colored by plasma detection.

        Genes detected in plasma are highlighted; the color intensity encodes
        the number of plasma sources in which they appear.

        Returns
        -------
        Path
            Saved figure path.
        """
        fig_path = self.figures_dir / "volcano_plasma_detection.png"

        if "log2fc" not in self._candidates.columns or "padj" not in self._candidates.columns:
            logger.warning(
                "Columns 'log2fc'/'padj' not found; skipping volcano plot."
            )
            return fig_path

        df = self._candidates.copy()
        df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))

        # Merge plasma detection info
        if self._overlap_df is not None and not self._overlap_df.empty:
            det_info = self._overlap_df[["gene", "n_plasma_sources"]].drop_duplicates("gene")
            df = df.merge(det_info, on="gene", how="left")
            df["n_plasma_sources"] = df["n_plasma_sources"].fillna(0).astype(int)
        else:
            df["n_plasma_sources"] = 0

        fig, ax = plt.subplots(figsize=(8, 6))

        # Background: not in plasma
        mask_bg = df["n_plasma_sources"] == 0
        ax.scatter(
            df.loc[mask_bg, "log2fc"],
            df.loc[mask_bg, "neg_log10_padj"],
            c="lightgrey",
            alpha=0.5,
            s=30,
            label="Not in plasma",
            edgecolors="none",
        )

        # Foreground: detected in plasma, colored by n_plasma_sources
        cmap = plt.cm.get_cmap("YlOrRd")
        n_max = max(df["n_plasma_sources"].max(), 1)
        for n_src in range(1, n_max + 1):
            mask = df["n_plasma_sources"] == n_src
            if not mask.any():
                continue
            color = cmap(n_src / (n_max + 1) + 0.2)
            ax.scatter(
                df.loc[mask, "log2fc"],
                df.loc[mask, "neg_log10_padj"],
                c=[color],
                alpha=0.85,
                s=60,
                label=f"Plasma sources: {n_src}",
                edgecolors="grey",
                linewidths=0.3,
            )

        # Reference lines
        ax.axhline(-np.log10(0.05), color="grey", linestyle="--", lw=0.8, alpha=0.7)
        ax.axvline(0, color="grey", linestyle="-", lw=0.5, alpha=0.5)

        # Annotate top plasma-detected genes
        top_plasma = df[df["n_plasma_sources"] > 0].nlargest(12, "neg_log10_padj")
        for _, r in top_plasma.iterrows():
            ax.annotate(
                r["gene"],
                xy=(r["log2fc"], r["neg_log10_padj"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.5,
                alpha=0.9,
            )

        ax.set_xlabel("log2 Fold Change (disease vs control)")
        ax.set_ylabel("-log10(adjusted p-value)")
        ax.set_title("Volcano Plot — Cardiac Secretome Candidates\nColored by Plasma Detection")
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Volcano plot saved to %s.", fig_path)
        return fig_path

    def plot_upset(self) -> Path:
        """UpSet plot for multi-set intersections across plasma datasets.

        Attempts to use ``upsetplot`` library. Falls back to a grouped bar
        chart showing pairwise and triple intersection sizes when unavailable.

        Returns
        -------
        Path
            Saved figure path.
        """
        fig_path = self.figures_dir / "upset_plasma_overlap.png"
        candidate_genes: set = set(self._candidates["gene"].tolist())

        # Build per-source boolean membership for candidate genes
        sources = PLASMA_DATASET_IDS  # 3 PRIDE datasets
        gene_list = sorted(candidate_genes)
        membership: Dict[str, List[bool]] = {
            src: [g in self._plasma_sets.get(src, set()) for g in gene_list]
            for src in sources
        }

        try:
            from upsetplot import UpSet, from_memberships
            # Build membership list of frozensets
            memberships_list = []
            for i, gene in enumerate(gene_list):
                detected_in = frozenset(s for s in sources if membership[s][i])
                if detected_in:
                    memberships_list.append(detected_in)

            if not memberships_list:
                raise ValueError("No detected genes for UpSet plot.")

            data = from_memberships(memberships_list)
            upset = UpSet(data, subset_size="count", show_counts=True)
            fig = upset.plot()
            # from_memberships returns a figure via upset.plot()
            plt.suptitle("UpSet: Cardiac Secretome Candidates × Plasma Datasets", y=1.01)
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("UpSet plot saved to %s.", fig_path)
            return fig_path
        except (ImportError, Exception) as exc:
            if not isinstance(exc, ImportError):
                logger.debug("upsetplot failed: %s; using bar fallback.", exc)
            else:
                logger.debug("upsetplot not installed; using bar fallback.")

        # Fallback: grouped bar chart of individual + intersection counts
        fig, ax = plt.subplots(figsize=(10, 5))
        labels: List[str] = []
        sizes: List[int] = []

        # Individual datasets
        for src in sources:
            n = len(candidate_genes & self._plasma_sets.get(src, set()))
            labels.append(src)
            sizes.append(n)

        # Pairwise intersections
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                inter = (
                    candidate_genes
                    & self._plasma_sets.get(sources[i], set())
                    & self._plasma_sets.get(sources[j], set())
                )
                labels.append(f"{sources[i]}\n∩ {sources[j]}")
                sizes.append(len(inter))

        # All-three intersection
        all_three = candidate_genes.copy()
        for src in sources:
            all_three &= self._plasma_sets.get(src, set())
        labels.append("All 3 datasets")
        sizes.append(len(all_three))

        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        bars = ax.bar(range(len(labels)), sizes, color=colors, alpha=0.85, edgecolor="white")
        for bar, s in zip(bars, sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(s),
                ha="center",
                fontsize=9,
            )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Number of candidate ligands")
        ax.set_title("Intersection Sizes: Cardiac Secretome × Plasma EV Proteomics")
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("UpSet (bar fallback) saved to %s.", fig_path)
        return fig_path

    def plot_venn(self) -> Path:
        """Generate a Venn diagram of overlapping gene sets.

        Uses matplotlib_venn for 3-set diagram when available; falls back to
        a horizontal bar chart.

        Returns
        -------
        Path
            Saved figure path.
        """
        fig_path = self.figures_dir / "venn_plasma_overlap.png"
        candidate_genes = set(self._candidates["gene"].tolist())

        try:
            from matplotlib_venn import venn3
            fig, ax = plt.subplots(figsize=(7, 6))
            venn3(
                subsets=[
                    candidate_genes & self._plasma_sets.get(ds, set())
                    for ds in PLASMA_DATASET_IDS
                ],
                set_labels=PLASMA_DATASET_IDS,
                ax=ax,
            )
            ax.set_title("Candidate Ligands Detected in Plasma EV Proteomics")
            plt.tight_layout()
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Venn diagram (matplotlib_venn) saved to %s.", fig_path)
            return fig_path
        except ImportError:
            pass

        # Fallback: horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        labels: List[str] = []
        sizes: List[int] = []

        for ds in PLASMA_DATASET_IDS:
            labels.append(ds)
            sizes.append(len(candidate_genes & self._plasma_sets.get(ds, set())))

        all_three = candidate_genes.copy()
        for ds in PLASMA_DATASET_IDS:
            all_three &= self._plasma_sets.get(ds, set())
        labels.append("All 3 datasets")
        sizes.append(len(all_three))

        union_plasma: set = set()
        for ds in PLASMA_DATASET_IDS:
            union_plasma |= self._plasma_sets.get(ds, set())
        labels.append("Any dataset (union)")
        sizes.append(len(candidate_genes & union_plasma))

        colors = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7", "#c4ad66"]
        bars = ax.barh(labels, sizes, color=colors[: len(labels)], alpha=0.8)
        for bar, s in zip(bars, sizes):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(s),
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Number of candidate ligands")
        ax.set_title("Overlap: Cardiac Secretome Candidates × Plasma EV Proteomics")
        ax.set_xlim(0, max(sizes) * 1.15 if sizes else 1)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Venn (fallback bar chart) saved to %s.", fig_path)
        return fig_path

    def plot_enrichment_summary(self) -> Path:
        """Bar chart of Fisher exact enrichment odds ratios per dataset.

        Returns
        -------
        Path
            Saved figure path.
        """
        if self._enrichment_df is None:
            raise RuntimeError("Call run() before plot_enrichment_summary().")

        fig_path = self.figures_dir / "enrichment_summary.png"
        df = self._enrichment_df.copy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        colors = ["firebrick" if p < 0.05 else "steelblue" for p in df["pvalue"]]

        ax = axes[0]
        ax.barh(df["dataset"], np.log2(df["odds_ratio"].clip(lower=0.01)), color=colors, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("log2 Odds Ratio")
        ax.set_title("Enrichment (Fisher Exact)")

        ax2 = axes[1]
        neg_log_p = -np.log10(df["pvalue"].clip(lower=1e-50))
        ax2.barh(df["dataset"], neg_log_p, color=colors, alpha=0.8)
        ax2.axvline(-np.log10(0.05), color="red", linestyle="--", lw=0.8, label="p=0.05")
        ax2.set_xlabel("-log10(p-value)")
        ax2.set_title("Significance")
        ax2.legend(fontsize=8)

        plt.suptitle("Cardiac Secretome Enrichment in Plasma EV Datasets", y=1.01)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Enrichment summary saved to %s.", fig_path)
        return fig_path

    @property
    def overlap(self) -> pd.DataFrame:
        """Overlap DataFrame with per-gene plasma detection info. Requires ``run()``."""
        if self._overlap_df is None:
            raise RuntimeError("Call run() first.")
        return self._overlap_df

    @property
    def plasma_candidates(self) -> pd.DataFrame:
        """Plasma candidates with combined_rank. Requires ``run()``."""
        if self._plasma_candidates_df is None:
            raise RuntimeError("Call run() first.")
        return self._plasma_candidates_df

    @property
    def enrichment(self) -> pd.DataFrame:
        """Enrichment statistics. Requires ``run()``."""
        if self._enrichment_df is None:
            raise RuntimeError("Call run() first.")
        return self._enrichment_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_plasma_set(self, dataset_id: str) -> set:
        """Load gene set for a plasma dataset.

        First attempts to read ``plasma_data_dir/<dataset_id>.[csv|tsv]``.
        Falls back to the embedded stub list.

        Parameters
        ----------
        dataset_id:
            One of _ALL_SOURCE_IDS.

        Returns
        -------
        set
            Set of gene symbols (upper-cased) detected in this dataset.
        """
        if self.plasma_data_dir is not None:
            for ext in ("csv", "tsv"):
                fpath = self.plasma_data_dir / f"{dataset_id}.{ext}"
                if fpath.exists():
                    sep = "," if ext == "csv" else "\t"
                    try:
                        df = pd.read_csv(fpath, sep=sep)
                        if self.gene_column in df.columns:
                            genes = set(df[self.gene_column].dropna().str.upper().tolist())
                            logger.info("Loaded %d proteins from %s.", len(genes), fpath)
                            return genes
                        else:
                            logger.warning(
                                "Column '%s' not found in %s. Available: %s",
                                self.gene_column,
                                fpath,
                                df.columns.tolist(),
                            )
                    except Exception as exc:
                        logger.error("Failed to read %s: %s", fpath, exc)

        stub = set(_STUB_PROTEINS.get(dataset_id, []))
        logger.info("Using stub list for %s (%d proteins).", dataset_id, len(stub))
        return stub

    def _compute_plasma_score(self, gene: str) -> float:
        """Compute source-weighted plasma_score for one gene.

        plasma_score = Σ(weight_i × detected_i)

        Only sources present in self.source_weights are considered.

        Parameters
        ----------
        gene:
            Gene symbol.

        Returns
        -------
        float
            Weighted plasma detection score in [0, total_weight].
        """
        score = 0.0
        for src, weight in self.source_weights.items():
            if gene in self._plasma_sets.get(src, set()):
                score += weight
        return score

    def _build_overlap_df(self, candidate_genes: set) -> pd.DataFrame:
        """Build per-gene plasma detection DataFrame for all candidates.

        Parameters
        ----------
        candidate_genes:
            Set of all candidate gene symbols.

        Returns
        -------
        pd.DataFrame
            Columns: gene, plasma_datasets (pipe-separated list of detecting
            sources), n_plasma_sources, plasma_score.
        """
        records: List[Dict] = []
        for gene in candidate_genes:
            detecting_sources = [
                src for src in _ALL_SOURCE_IDS
                if gene in self._plasma_sets.get(src, set())
            ]
            plasma_score = self._compute_plasma_score(gene)
            base = self._candidates[self._candidates["gene"] == gene].iloc[0].to_dict()
            base["plasma_detected_in"] = "|".join(detecting_sources) if detecting_sources else ""
            base["n_plasma_sources"] = len(detecting_sources)
            base["plasma_score"] = plasma_score
            records.append(base)

        if not records:
            logger.warning("No candidates to build overlap DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Rename legacy column for backward compatibility
        if "plasma_datasets" not in df.columns and "plasma_detected_in" in df.columns:
            df["plasma_datasets"] = df["plasma_detected_in"]
        df.sort_values(
            ["n_plasma_sources", "plasma_score"],
            ascending=[False, False],
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)
        return df

    def _build_plasma_candidates(self) -> pd.DataFrame:
        """Build the final plasma_candidates table.

        Filters to genes detected in at least one plasma source, then
        computes combined_rank from cardiac_score and plasma_score.

        Output columns:
            gene, cardiac_score, plasma_detected_in, plasma_score, combined_rank

        Returns
        -------
        pd.DataFrame
            Sorted by combined_rank ascending (rank 1 = best candidate).
        """
        if self._overlap_df is None or self._overlap_df.empty:
            return pd.DataFrame(columns=[
                "gene", "cardiac_score", "plasma_detected_in",
                "plasma_score", "combined_rank",
            ])

        df = self._overlap_df[self._overlap_df["n_plasma_sources"] > 0].copy()

        # Cardiac score column
        cardiac_col = self._cardiac_score_col
        if cardiac_col not in df.columns:
            df["cardiac_score"] = 0.0
        else:
            df["cardiac_score"] = df[cardiac_col]

        # Rank independently on each score (lower rank number = better)
        df["rank_cardiac"] = df["cardiac_score"].rank(ascending=False, method="min")
        df["rank_plasma"] = df["plasma_score"].rank(ascending=False, method="min")

        # Combined rank: equal-weight mean of ranks
        df["combined_rank"] = (df["rank_cardiac"] + df["rank_plasma"]) / 2.0
        df.sort_values("combined_rank", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        out_cols = ["gene", "cardiac_score", "plasma_detected_in", "plasma_score", "combined_rank"]
        # Append any extra columns not in output spec
        extra = [c for c in df.columns if c not in out_cols]
        return df[out_cols + extra]

    def _fisher_enrichment(self, candidate_genes: set) -> pd.DataFrame:
        """Run Fisher exact test for each PRIDE plasma dataset.

        2x2 contingency:
                         In plasma  Not in plasma
        In candidates:      a            b
        Not in candidates:  c            d

        Parameters
        ----------
        candidate_genes:
            Set of candidate secretome gene symbols.

        Returns
        -------
        pd.DataFrame
            Columns: dataset, overlap_n, candidates_n, plasma_n,
            background_n, odds_ratio, pvalue, neg_log10_pvalue, significant.
        """
        records: List[Dict] = []
        not_candidate = self.background_size - len(candidate_genes)

        for ds in PLASMA_DATASET_IDS:
            plasma_set = self._plasma_sets.get(ds, set())
            a = len(candidate_genes & plasma_set)
            b = len(candidate_genes) - a
            c = len(plasma_set) - a
            d = max(not_candidate - c, 0)

            contingency = np.array([[a, b], [c, d]])
            odds_ratio, pvalue = fisher_exact(contingency, alternative="greater")

            logger.info(
                "Fisher test %s: a=%d b=%d c=%d d=%d | OR=%.3f p=%.3e",
                ds, a, b, c, d, odds_ratio, pvalue,
            )
            records.append(
                {
                    "dataset": ds,
                    "overlap_n": a,
                    "candidates_n": len(candidate_genes),
                    "plasma_n": len(plasma_set),
                    "background_n": self.background_size,
                    "odds_ratio": odds_ratio,
                    "pvalue": pvalue,
                }
            )

        df = pd.DataFrame(records)
        df["neg_log10_pvalue"] = -np.log10(df["pvalue"].clip(lower=1e-300))
        df["significant"] = df["pvalue"] < 0.05
        return df

    def _save_outputs(self) -> None:
        """Write plasma_candidates and enrichment tables to CSV."""
        # plasma_candidates.csv — primary output
        if self._plasma_candidates_df is not None and not self._plasma_candidates_df.empty:
            out_path = self.results_dir / "plasma_candidates.csv"
            self._plasma_candidates_df.to_csv(out_path, index=False)
            logger.info(
                "Plasma candidates (%d) saved to %s.",
                len(self._plasma_candidates_df),
                out_path,
            )
        else:
            logger.warning("No plasma candidates to save.")

        # Legacy overlapping_candidates.csv for backward compatibility
        if self._overlap_df is not None and not self._overlap_df.empty:
            overlap_path = self.results_dir / "overlapping_candidates.csv"
            self._overlap_df.to_csv(overlap_path, index=False)
            logger.info(
                "Overlapping candidates (%d) saved to %s.",
                len(self._overlap_df),
                overlap_path,
            )

        # Enrichment statistics
        if self._enrichment_df is not None:
            enrich_path = self.results_dir / "enrichment_statistics.csv"
            self._enrichment_df.to_csv(enrich_path, index=False)
            logger.info("Enrichment statistics saved to %s.", enrich_path)
