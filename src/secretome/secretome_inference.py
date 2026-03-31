"""
Cardiac secretome inference from single-cell DEG results.

Workflow
--------
1. Load curated secretome gene lists (signal-peptide annotated genes and
   GO:0005615 extracellular space genes).
2. Cross with DEGs from disease fibroblasts and cardiomyocytes.
3. Score EV-relevant markers (tetraspanins + biogenesis proteins).
4. Compute multi-feature composite score per gene candidate.
5. Export candidate_ligands.csv with all features and composite score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated marker collections
# ---------------------------------------------------------------------------

# Extracellular vesicle surface / biogenesis markers
EV_MARKERS: List[str] = [
    "CD9",
    "CD63",
    "CD81",
    "ALIX",   # PDCD6IP
    "TSG101",
    "FLOT1",
    "FLOT2",
    "RAB27A",
    "RAB27B",
    "SYNTENIN1",  # SDCBP
]

# Minimal curated secretome seed — signal-peptide carrying cardiac-relevant
# genes (subset; augmented at runtime from provided gene lists).
_SIGNAL_PEPTIDE_SEED: List[str] = [
    "IL6", "TNF", "TGFB1", "TGFB2", "TGFB3",
    "VEGFA", "VEGFB", "VEGFC",
    "FGF1", "FGF2", "FGF7", "FGF10", "FGF21",
    "PDGFA", "PDGFB", "PDGFC", "PDGFD",
    "IGF1", "IGF2",
    "BMP2", "BMP4", "BMP6", "BMP7",
    "WNT5A", "WNT3A", "WNT11",
    "DKK1", "DKK3",
    "SFRP1", "SFRP2", "SFRP3",
    "ANGPT1", "ANGPT2",
    "CXCL12", "CXCL1", "CXCL8",
    "CCL2", "CCL3", "CCL5",
    "COL1A1", "COL1A2", "COL3A1", "COL4A1",
    "FN1", "POSTN", "CTGF",
    "MMP1", "MMP2", "MMP3", "MMP9", "MMP14",
    "TIMP1", "TIMP2",
    "SERPINE1", "SERPINE2",
    "NPPA", "NPPB",
    "GDF15", "GDF11", "GDF8",
    "MSTN",
    "HBEGF", "EGF", "EPIREGULIN", "AREG",
    "EREG",
    "NRG1", "NRG2",
    "NOTCH1", "JAG1", "DLL1",
    "ENG", "ACVRL1",
]

# GO:0005615 extracellular space seed (curated cardiac subset)
_GO_EXTRACELLULAR_SEED: List[str] = [
    "THBS1", "THBS2", "THBS4",
    "COMP", "MATN2", "MATN3",
    "LTBP1", "LTBP2", "LTBP3", "LTBP4",
    "LOXL1", "LOXL2", "LOXL3",
    "ADAMTS1", "ADAMTS2", "ADAMTS5",
    "FBLN1", "FBLN2", "FBLN5",
    "TNC", "TNXB",
    "VTN", "VWF",
    "HSPG2", "VCAN", "ACAN",
    "SPP1",  # osteopontin
    "BGN", "DCN", "FMOD", "LUM",
    "CILP", "CILP2",
    "CXCL16", "CX3CL1",
    "GAS6", "PROS1",
    "PENK", "SCG2", "VGF",
    "CLU", "APOE", "APOC1",
    "LGALS1", "LGALS3",
    "HMGB1",
    "S100A8", "S100A9",
    "RETN",
]

# Combined secreted/ECM gene set for extracellular_flag
_SECRETED_ECM_GENES: Set[str] = set(_SIGNAL_PEPTIDE_SEED) | set(_GO_EXTRACELLULAR_SEED)


class CardiacSecretomeInference:
    """Infer candidate cardiac secreted ligands from DEG tables.

    Multi-feature scoring per gene:

    - expression_score       : mean expression in disease cells (normalized)
    - prevalence_score       : fraction of cells expressing the gene
    - extracellular_flag     : 1 if gene is in secreted/ECM gene set
    - ev_flag                : 1 if gene is in EV-related gene set
    - disease_logFC          : log fold change disease vs control
    - cell_type_specificity  : max expression ratio across cell types
    - cross_disease_presence : detected in multiple disease conditions

    Final composite score:
        secretome_score = 0.25*expression + 0.15*prevalence + 0.15*extracellular
                        + 0.15*ev + 0.15*logFC + 0.15*specificity

    Parameters
    ----------
    deg_tables:
        Mapping of cell_type -> DataFrame with at least columns
        ``gene``, ``log2fc``, ``padj``.  Optionally includes ``mean_expr``
        (mean expression in disease cells) and ``pct_expr`` (fraction of
        cells expressing). Typically the output of
        ``DifferentialExpression.run()``.
    cell_types_of_interest:
        Subset of cell types to use for secretome inference.
        Defaults to ``["fibroblast", "cardiomyocyte"]``.
    disease_conditions:
        Mapping of condition_name -> set of cell_type keys belonging to that
        condition. Used to compute cross_disease_presence. If None, each
        cell type in deg_tables is treated as a separate condition.
    logfc_threshold:
        Minimum |log2FC| for a gene to be considered a DEG.
    padj_threshold:
        Maximum adjusted p-value for a gene to be considered a DEG.
    extra_secretome_genes:
        Additional gene names to append to the curated secretome list.
    figures_dir:
        Output directory for figures.
    results_dir:
        Output directory for CSV tables.
    """

    def __init__(
        self,
        deg_tables: Dict[str, pd.DataFrame],
        cell_types_of_interest: Optional[List[str]] = None,
        disease_conditions: Optional[Dict[str, List[str]]] = None,
        logfc_threshold: float = 0.5,
        padj_threshold: float = 0.05,
        extra_secretome_genes: Optional[List[str]] = None,
        figures_dir: Path = Path("results/figures"),
        results_dir: Path = Path("results"),
    ) -> None:
        self.deg_tables = deg_tables
        self.cell_types_of_interest: List[str] = (
            cell_types_of_interest
            if cell_types_of_interest is not None
            else ["fibroblast", "cardiomyocyte"]
        )
        self.disease_conditions = disease_conditions
        self.logfc_threshold = logfc_threshold
        self.padj_threshold = padj_threshold
        self.figures_dir = Path(figures_dir)
        self.results_dir = Path(results_dir)

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Build the full secretome reference set
        self._secretome_genes: Set[str] = self._build_secretome_set(
            extra_secretome_genes or []
        )
        # Populated after run()
        self._candidates: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Perform full secretome inference pipeline.

        Returns
        -------
        pd.DataFrame
            Ranked candidate ligands with all feature scores and composite score.
        """
        logger.info(
            "Secretome inference — %d secretome genes, cell types: %s",
            len(self._secretome_genes),
            self.cell_types_of_interest,
        )

        records: List[Dict] = []

        for ct in self.cell_types_of_interest:
            matched_key = self._match_cell_type_key(ct)
            if matched_key is None:
                logger.warning(
                    "Cell type '%s' not found in deg_tables. Available: %s",
                    ct,
                    list(self.deg_tables.keys()),
                )
                continue

            deg_df = self.deg_tables[matched_key]
            sig_up = self._filter_significant(deg_df, direction="up")
            overlap = sig_up[sig_up["gene"].isin(self._secretome_genes)].copy()

            logger.info(
                "  %s: %d sig-up DEGs, %d in secretome reference.",
                matched_key,
                len(sig_up),
                len(overlap),
            )

            for _, row in overlap.iterrows():
                gene = row["gene"]
                expr_score = self._get_expression_score(gene, deg_df)
                prevalence = self._get_prevalence_score(gene, deg_df)
                records.append(
                    {
                        "gene": gene,
                        "cell_type": matched_key,
                        # Multi-feature columns
                        "expression_score": expr_score,
                        "prevalence_score": prevalence,
                        "extracellular_flag": int(gene in _SECRETED_ECM_GENES),
                        "ev_flag": int(gene in set(EV_MARKERS)),
                        "disease_logFC": float(row["log2fc"]),
                        # cell_type_specificity and cross_disease_presence
                        # are computed post-aggregation
                        # Legacy columns retained for backward compatibility
                        "log2fc": float(row["log2fc"]),
                        "padj": float(row["padj"]),
                        "in_signal_peptide_set": gene in set(_SIGNAL_PEPTIDE_SEED),
                        "in_go_extracellular_set": gene in set(_GO_EXTRACELLULAR_SEED),
                        "is_ev_marker": gene in set(EV_MARKERS),
                        "ev_coexpression_score": self._compute_ev_score(gene, deg_df),
                    }
                )

        if not records:
            logger.warning("No candidate ligands found. Check DEG thresholds.")
            self._candidates = pd.DataFrame(columns=[
                "gene", "cell_type",
                "expression_score", "prevalence_score",
                "extracellular_flag", "ev_flag",
                "disease_logFC", "cell_type_specificity", "cross_disease_presence",
                "log2fc", "padj",
                "in_signal_peptide_set", "in_go_extracellular_set",
                "is_ev_marker", "ev_coexpression_score",
                "secretome_score",
            ])
            return self._candidates

        candidates = pd.DataFrame(records)

        # Compute cross-table features before deduplication
        candidates = self._add_cell_type_specificity(candidates)
        candidates = self._add_cross_disease_presence(candidates)

        candidates = self._deduplicate_across_cell_types(candidates)

        # Composite score with new formula
        candidates["secretome_score"] = self._secretome_composite_score(candidates)
        # Legacy composite_score kept for downstream compatibility
        candidates["composite_score"] = candidates["secretome_score"]

        candidates.sort_values("secretome_score", ascending=False, inplace=True)
        candidates.reset_index(drop=True, inplace=True)

        self._candidates = candidates
        self._save_candidates()
        logger.info(
            "Secretome inference complete: %d candidate ligands.", len(candidates)
        )
        return candidates

    def plot_ev_marker_scores(self) -> Path:
        """Bar chart of EV marker expression across cell types of interest.

        Returns
        -------
        Path
            Saved figure path.
        """
        fig_path = self.figures_dir / "ev_marker_scores.png"

        plot_data: List[Dict] = []
        for ct in self.cell_types_of_interest:
            matched = self._match_cell_type_key(ct)
            if matched is None:
                continue
            deg_df = self.deg_tables[matched]
            for marker in EV_MARKERS:
                row = deg_df[deg_df["gene"] == marker]
                lfc = float(row["log2fc"].values[0]) if len(row) else 0.0
                plot_data.append({"marker": marker, "cell_type": matched, "log2fc": lfc})

        if not plot_data:
            logger.warning("No EV marker data to plot.")
            return fig_path

        df = pd.DataFrame(plot_data)
        cell_types = df["cell_type"].unique().tolist()
        markers = EV_MARKERS
        x = np.arange(len(markers))
        width = 0.8 / max(len(cell_types), 1)

        fig, ax = plt.subplots(figsize=(max(8, len(markers) * 0.9), 5))
        for i, ct in enumerate(cell_types):
            sub = df[df["cell_type"] == ct].set_index("marker")
            vals = [sub.loc[m, "log2fc"] if m in sub.index else 0.0 for m in markers]
            ax.bar(x + i * width, vals, width=width, label=ct, alpha=0.8)

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x + width * (len(cell_types) - 1) / 2)
        ax.set_xticklabels(markers, rotation=45, ha="right")
        ax.set_ylabel("log2 Fold Change (disease vs control)")
        ax.set_title("EV Marker Expression in Disease Cells")
        ax.legend()
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("EV marker figure saved to %s.", fig_path)
        return fig_path

    def plot_feature_scatter(self) -> Path:
        """Scatter plot of disease_logFC vs expression_score, colored by secretome_score.

        Returns
        -------
        Path
            Saved figure path.
        """
        if self._candidates is None:
            raise RuntimeError("Call run() first.")

        fig_path = self.figures_dir / "secretome_feature_scatter.png"
        df = self._candidates.copy()

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            df["disease_logFC"],
            df["expression_score"],
            c=df["secretome_score"],
            cmap="YlOrRd",
            alpha=0.8,
            edgecolors="grey",
            linewidths=0.3,
            s=60,
        )
        plt.colorbar(sc, ax=ax, label="Secretome Score")

        # Annotate top 10 genes
        top = df.nlargest(10, "secretome_score")
        for _, r in top.iterrows():
            ax.annotate(
                r["gene"],
                xy=(r["disease_logFC"], r["expression_score"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        ax.axvline(self.logfc_threshold, color="grey", linestyle="--", lw=0.8)
        ax.set_xlabel("Disease logFC")
        ax.set_ylabel("Expression Score (normalized)")
        ax.set_title("Secretome Candidates: logFC vs Expression")
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature scatter saved to %s.", fig_path)
        return fig_path

    @property
    def candidates(self) -> pd.DataFrame:
        """Ranked candidate ligands. Requires ``run()`` to be called first."""
        if self._candidates is None:
            raise RuntimeError("Call run() first.")
        return self._candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_secretome_set(extra: List[str]) -> Set[str]:
        genes = set(_SIGNAL_PEPTIDE_SEED) | set(_GO_EXTRACELLULAR_SEED) | set(extra)
        logger.debug("Secretome reference built: %d unique genes.", len(genes))
        return genes

    def _filter_significant(
        self, df: pd.DataFrame, direction: str = "up"
    ) -> pd.DataFrame:
        mask = df["padj"] < self.padj_threshold
        if direction == "up":
            mask &= df["log2fc"] >= self.logfc_threshold
        elif direction == "down":
            mask &= df["log2fc"] <= -self.logfc_threshold
        return df[mask].copy()

    def _match_cell_type_key(self, cell_type: str) -> Optional[str]:
        """Case-insensitive partial match against deg_tables keys."""
        ct_lower = cell_type.lower()
        for key in self.deg_tables:
            if ct_lower in key.lower():
                return key
        return None

    @staticmethod
    def _get_expression_score(gene: str, deg_df: pd.DataFrame) -> float:
        """Return mean expression score for the gene if available, else use log2fc proxy.

        If ``mean_expr`` column exists in deg_df, uses that value directly.
        Otherwise falls back to the absolute log2fc value as a proxy.

        Parameters
        ----------
        gene:
            Gene symbol.
        deg_df:
            DEG table for a cell type.

        Returns
        -------
        float
            Expression score (non-negative).
        """
        row = deg_df[deg_df["gene"] == gene]
        if row.empty:
            return 0.0
        if "mean_expr" in deg_df.columns:
            val = row["mean_expr"].values[0]
            return float(val) if not np.isnan(val) else 0.0
        # Proxy: absolute log2fc (already positive because we filter sig-up)
        return float(abs(row["log2fc"].values[0]))

    @staticmethod
    def _get_prevalence_score(gene: str, deg_df: pd.DataFrame) -> float:
        """Return fraction of cells expressing the gene.

        Uses ``pct_expr`` column if present; otherwise returns 0.5 as neutral
        default to avoid penalizing genes without this information.

        Parameters
        ----------
        gene:
            Gene symbol.
        deg_df:
            DEG table for a cell type.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if "pct_expr" not in deg_df.columns:
            return 0.5
        row = deg_df[deg_df["gene"] == gene]
        if row.empty:
            return 0.0
        val = row["pct_expr"].values[0]
        # pct_expr may be stored as 0-100 or 0-1; normalize to [0,1]
        val = float(val) if not np.isnan(val) else 0.5
        return val / 100.0 if val > 1.0 else val

    def _compute_ev_score(self, gene: str, deg_df: pd.DataFrame) -> float:
        """Score how EV-associated a gene is.

        Returns 1.0 if the gene is an EV marker, 0.5 if any EV marker is
        significantly up-regulated in the same cell type, else 0.0.

        Parameters
        ----------
        gene:
            Gene symbol.
        deg_df:
            DEG table for a cell type.

        Returns
        -------
        float
            EV co-expression score in {0.0, 0.5, 1.0}.
        """
        if gene in set(EV_MARKERS):
            return 1.0
        sig_up = self._filter_significant(deg_df, direction="up")
        n_ev_up = sig_up["gene"].isin(set(EV_MARKERS)).sum()
        if n_ev_up > 0:
            return 0.5
        return 0.0

    def _add_cell_type_specificity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute max expression ratio across cell types.

        For each gene, the specificity is the ratio of its expression score
        in its highest-expressing cell type to the mean across all cell types
        in which it appears. Genes appearing in only one cell type get a score
        of 1.0 (maximally specific).

        Parameters
        ----------
        df:
            Records DataFrame before deduplication.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``cell_type_specificity`` column added.
        """
        gene_ct_expr = (
            df.groupby(["gene", "cell_type"])["expression_score"]
            .mean()
            .reset_index()
        )
        specificity_map: Dict[str, float] = {}
        for gene, grp in gene_ct_expr.groupby("gene"):
            vals = grp["expression_score"].values.astype(float)
            mean_val = vals.mean()
            max_val = vals.max()
            if mean_val > 1e-9:
                specificity_map[str(gene)] = float(max_val / mean_val)
            else:
                specificity_map[str(gene)] = 1.0

        df = df.copy()
        df["cell_type_specificity"] = df["gene"].map(specificity_map).fillna(1.0)
        return df

    def _add_cross_disease_presence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count how many distinct disease conditions each gene appears in.

        If ``disease_conditions`` was provided, uses that mapping. Otherwise,
        each unique cell_type key is treated as a separate condition.

        The raw count is normalized to [0, 1] by dividing by the total number
        of conditions.

        Parameters
        ----------
        df:
            Records DataFrame.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``cross_disease_presence`` column added.
        """
        df = df.copy()

        if self.disease_conditions:
            # Build gene -> set of condition names
            gene_conditions: Dict[str, Set[str]] = {}
            for gene in df["gene"].unique():
                gene_cts = set(df[df["gene"] == gene]["cell_type"].tolist())
                conditions_found: Set[str] = set()
                for cond_name, cond_cts in self.disease_conditions.items():
                    if gene_cts & set(cond_cts):
                        conditions_found.add(cond_name)
                gene_conditions[gene] = conditions_found
            n_total = max(len(self.disease_conditions), 1)
        else:
            # Each cell type is its own condition
            gene_conditions = {
                gene: set(sub["cell_type"].tolist())
                for gene, sub in df.groupby("gene")
            }
            n_total = max(df["cell_type"].nunique(), 1)

        df["cross_disease_presence"] = df["gene"].map(
            lambda g: len(gene_conditions.get(g, set())) / n_total
        )
        return df

    @staticmethod
    def _deduplicate_across_cell_types(df: pd.DataFrame) -> pd.DataFrame:
        """For genes appearing in multiple cell types, keep the row with highest log2fc."""
        idx = df.groupby("gene")["log2fc"].idxmax()
        deduped = df.loc[idx].copy()
        # Annotate all source cell types
        source_map = (
            df.groupby("gene")["cell_type"]
            .apply(lambda x: "|".join(sorted(x.unique())))
            .to_dict()
        )
        deduped["source_cell_types"] = deduped["gene"].map(source_map)
        return deduped

    @staticmethod
    def _min_max_norm(series: pd.Series) -> pd.Series:
        """Min-max normalize a Series to [0, 1].

        Parameters
        ----------
        series:
            Numeric pandas Series.

        Returns
        -------
        pd.Series
            Normalized Series.
        """
        lo, hi = series.min(), series.max()
        if hi - lo < 1e-9:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - lo) / (hi - lo)

    def _secretome_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute weighted composite secretome score.

        Formula:
            secretome_score = 0.25 * expression_score_norm
                            + 0.15 * prevalence_score_norm
                            + 0.15 * extracellular_flag
                            + 0.15 * ev_flag
                            + 0.15 * logFC_norm
                            + 0.15 * specificity_norm

        cross_disease_presence is incorporated as an additive bonus
        (multiplied by 0.10) on top of the weighted sum, then clipped to [0,1].

        Parameters
        ----------
        df:
            Deduplicated candidates DataFrame containing all feature columns.

        Returns
        -------
        pd.Series
            Composite score in approximately [0, 1].
        """
        expr_norm = self._min_max_norm(df["expression_score"])
        prev_norm = self._min_max_norm(df["prevalence_score"])
        ext_flag = df["extracellular_flag"].astype(float)
        ev_flag = df["ev_flag"].astype(float)
        logfc_norm = self._min_max_norm(df["disease_logFC"])
        spec_norm = self._min_max_norm(df["cell_type_specificity"])
        cross = df["cross_disease_presence"].astype(float)

        score = (
            0.25 * expr_norm
            + 0.15 * prev_norm
            + 0.15 * ext_flag
            + 0.15 * ev_flag
            + 0.15 * logfc_norm
            + 0.15 * spec_norm
            + 0.10 * cross
        )
        return score.clip(upper=1.0)

    def _save_candidates(self) -> None:
        """Write enhanced candidate_ligands.csv with all features and composite score."""
        out_path = self.results_dir / "candidate_ligands.csv"
        assert self._candidates is not None

        # Ordered column list — multi-feature columns first, legacy columns last
        preferred_cols = [
            "gene", "cell_type", "source_cell_types",
            "expression_score", "prevalence_score",
            "extracellular_flag", "ev_flag",
            "disease_logFC", "cell_type_specificity", "cross_disease_presence",
            "secretome_score",
            # legacy / supplementary
            "log2fc", "padj",
            "in_signal_peptide_set", "in_go_extracellular_set",
            "is_ev_marker", "ev_coexpression_score",
            "composite_score",
        ]
        cols = [c for c in preferred_cols if c in self._candidates.columns]
        extra = [c for c in self._candidates.columns if c not in cols]
        self._candidates[cols + extra].to_csv(out_path, index=False)
        logger.info(
            "Candidate ligands (%d) saved to %s.",
            len(self._candidates),
            out_path,
        )
