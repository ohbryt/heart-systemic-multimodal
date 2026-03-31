"""
Cardiac secretome inference from single-cell DEG results.

Workflow
--------
1. Load curated secretome gene lists (signal-peptide annotated genes and
   GO:0005615 extracellular space genes).
2. Cross with DEGs from disease fibroblasts and cardiomyocytes.
3. Score EV-relevant markers (tetraspanins + biogenesis proteins).
4. Rank candidate secreted ligands by a composite score.
5. Export candidate_ligands.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

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


class CardiacSecretomeInference:
    """Infer candidate cardiac secreted ligands from DEG tables.

    Parameters
    ----------
    deg_tables:
        Mapping of cell_type -> DataFrame with at least columns
        ``gene``, ``log2fc``, ``padj``.  Typically the output of
        ``DifferentialExpression.run()``.
    cell_types_of_interest:
        Subset of cell types to use for secretome inference.
        Defaults to ``["fibroblast", "cardiomyocyte"]``.
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
            Ranked candidate ligands with source annotations.
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
                ev_score = self._compute_ev_score(gene, deg_df)
                records.append(
                    {
                        "gene": gene,
                        "cell_type": matched_key,
                        "log2fc": row["log2fc"],
                        "padj": row["padj"],
                        "in_signal_peptide_set": gene in set(_SIGNAL_PEPTIDE_SEED),
                        "in_go_extracellular_set": gene in set(_GO_EXTRACELLULAR_SEED),
                        "is_ev_marker": gene in set(EV_MARKERS),
                        "ev_coexpression_score": ev_score,
                    }
                )

        if not records:
            logger.warning("No candidate ligands found. Check DEG thresholds.")
            self._candidates = pd.DataFrame(columns=[
                "gene", "cell_type", "log2fc", "padj",
                "in_signal_peptide_set", "in_go_extracellular_set",
                "is_ev_marker", "ev_coexpression_score", "composite_score",
            ])
            return self._candidates

        candidates = pd.DataFrame(records)
        candidates = self._deduplicate_across_cell_types(candidates)
        candidates["composite_score"] = self._composite_score(candidates)
        candidates.sort_values("composite_score", ascending=False, inplace=True)
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

    def _compute_ev_score(self, gene: str, deg_df: pd.DataFrame) -> float:
        """Score how EV-associated a gene is.

        Currently uses a simple rule: 1.0 if the gene itself is an EV marker,
        else 0.5 if any EV marker is significantly up-regulated in the same
        cell type (indicating active EV secretion), else 0.0.
        """
        if gene in set(EV_MARKERS):
            return 1.0
        sig_up = self._filter_significant(deg_df, direction="up")
        n_ev_up = sig_up["gene"].isin(set(EV_MARKERS)).sum()
        if n_ev_up > 0:
            return 0.5
        return 0.0

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
    def _composite_score(df: pd.DataFrame) -> pd.Series:
        """Weighted composite score for candidate ranking.

        Components
        ----------
        - log2fc                     (weight 0.4)
        - -log10(padj)               (weight 0.3)
        - in_signal_peptide_set bool (weight 0.15)
        - ev_coexpression_score      (weight 0.15)
        """
        lfc_norm = (df["log2fc"] - df["log2fc"].min()) / (
            df["log2fc"].max() - df["log2fc"].min() + 1e-9
        )
        neg_log_padj = -np.log10(df["padj"].clip(lower=1e-300))
        padj_norm = (neg_log_padj - neg_log_padj.min()) / (
            neg_log_padj.max() - neg_log_padj.min() + 1e-9
        )
        sp_bool = df["in_signal_peptide_set"].astype(float)
        ev_score = df["ev_coexpression_score"]

        return 0.4 * lfc_norm + 0.3 * padj_norm + 0.15 * sp_bool + 0.15 * ev_score

    def _save_candidates(self) -> None:
        out_path = self.results_dir / "candidate_ligands.csv"
        assert self._candidates is not None
        self._candidates.to_csv(out_path, index=False)
        logger.info(
            "Candidate ligands (%d) saved to %s.",
            len(self._candidates),
            out_path,
        )
