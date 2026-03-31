"""
Multi-criteria biomarker ranking module.

Scores each candidate by:
    cardiac_expression × secretome_score × plasma_detection × receptor_coverage

Produces ranked tables and publication-ready forest / lollipop plots.

Output files
------------
results/tables/ranked_biomarkers.csv
results/tables/ranked_lr_axes.csv
results/figures/ranked_biomarkers_lollipop.pdf/.png
results/figures/ranked_biomarkers_forest.pdf/.png

Usage
-----
    ranker = BiomarkerRanker(output_dir=Path("results"))
    df = ranker.rank(candidates_df, lr_pairs_df)
    ranker.save(df)
    ranker.plot(df)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default scoring weights — all must sum to 1.0
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS: dict[str, float] = {
    "cardiac_expression": 0.30,
    "secretome_score": 0.25,
    "plasma_detection": 0.25,
    "receptor_coverage": 0.20,
}

_FIGURE_DPI = 300
_FIGURE_STYLE = "whitegrid"


@dataclass
class RankingWeights:
    """
    Configurable weights for multi-criteria biomarker scoring.

    All weights must sum to 1.0 (validated on instantiation).
    """

    cardiac_expression: float = 0.30
    secretome_score: float = 0.25
    plasma_detection: float = 0.25
    receptor_coverage: float = 0.20

    def __post_init__(self) -> None:
        total = (
            self.cardiac_expression
            + self.secretome_score
            + self.plasma_detection
            + self.receptor_coverage
        )
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"RankingWeights must sum to 1.0, got {total:.6f}. "
                "Adjust the four weight fields accordingly."
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "cardiac_expression": self.cardiac_expression,
            "secretome_score": self.secretome_score,
            "plasma_detection": self.plasma_detection,
            "receptor_coverage": self.receptor_coverage,
        }


class BiomarkerRanker:
    """
    Rank cardiac biomarker candidates using four evidence dimensions.

    Parameters
    ----------
    output_dir : Path
        Root results directory.  Tables go to ``output_dir/tables``,
        figures to ``output_dir/figures``.
    weights : RankingWeights, optional
        Scoring weights.  Defaults to equal-ish preset.
    top_n : int
        How many top candidates to highlight in plots (default 20).
    figure_formats : list[str]
        Output formats for figures (default: ["pdf", "png"]).
    """

    def __init__(
        self,
        output_dir: Path = Path("results"),
        weights: Optional[RankingWeights] = None,
        top_n: int = 20,
        figure_formats: Optional[list[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self._tables_dir = self.output_dir / "tables"
        self._figures_dir = self.output_dir / "figures"
        self._tables_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir.mkdir(parents=True, exist_ok=True)

        self.weights = weights if weights is not None else RankingWeights()
        self.top_n = top_n
        self.figure_formats = figure_formats if figure_formats is not None else ["pdf", "png"]

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_candidates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure candidates DataFrame has required columns with sensible dtypes.

        Expected columns (all numeric, range 0–1 unless noted):
            gene            – gene symbol (str)
            cardiac_expression – mean/normalised expression in cardiac cells
            secretome_score – probability/score of being secreted
            plasma_detection   – detection score in plasma (proteomics/EV)

        Missing optional columns are filled with 0.5 (neutral).
        """
        required = ["gene"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"candidates DataFrame missing required column: '{col}'")

        score_cols = ["cardiac_expression", "secretome_score", "plasma_detection"]
        for col in score_cols:
            if col not in df.columns:
                logger.warning("Column '%s' not found — filling with 0.5", col)
                df = df.copy()
                df[col] = 0.5

        # Clip to [0, 1]
        for col in score_cols:
            df[col] = df[col].clip(0.0, 1.0)

        return df.drop_duplicates(subset=["gene"]).reset_index(drop=True)

    @staticmethod
    def _compute_receptor_coverage(
        candidates: pd.DataFrame,
        lr_pairs: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Compute receptor_coverage score per gene from LR pairs table.

        Score = (number of unique target tissues with ≥1 receptor hit) /
                (max tissue count across all candidates).
        Normalised to [0, 1].
        """
        if lr_pairs is None or lr_pairs.empty:
            logger.warning("No LR pairs provided — receptor_coverage set to 0.0")
            return pd.Series(0.0, index=candidates["gene"])

        # Count unique target tissues per ligand
        tissue_counts = (
            lr_pairs.groupby("ligand")["target_tissue"]
            .nunique()
            .rename("tissue_count")
        )
        max_count = tissue_counts.max() if not tissue_counts.empty else 1
        normalised = (tissue_counts / max_count).clip(0.0, 1.0)

        # Also weight by mean LR score per ligand
        mean_score = lr_pairs.groupby("ligand")["score"].mean().rename("mean_lr_score")
        combined = (0.6 * normalised + 0.4 * mean_score).clip(0.0, 1.0)

        return candidates["gene"].map(combined).fillna(0.0)

    # ------------------------------------------------------------------
    # Core ranking
    # ------------------------------------------------------------------

    def rank(
        self,
        candidates: pd.DataFrame,
        lr_pairs: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rank biomarker candidates by multi-criteria composite score.

        Parameters
        ----------
        candidates : pd.DataFrame
            Columns: gene, cardiac_expression, secretome_score, plasma_detection.
            Additional columns are preserved in output.
        lr_pairs : pd.DataFrame, optional
            Output of CrossOrganMapper (lr_pairs.csv).
            Used to compute receptor_coverage.

        Returns
        -------
        pd.DataFrame
            Input columns plus: receptor_coverage, composite_score, rank,
            evidence_summary.  Sorted by rank ascending.
        """
        df = self._validate_candidates(candidates.copy())
        w = self.weights

        # Receptor coverage from LR pairs
        df["receptor_coverage"] = self._compute_receptor_coverage(df, lr_pairs)

        # Weighted composite score
        df["composite_score"] = (
            w.cardiac_expression * df["cardiac_expression"]
            + w.secretome_score * df["secretome_score"]
            + w.plasma_detection * df["plasma_detection"]
            + w.receptor_coverage * df["receptor_coverage"]
        ).round(4)

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)

        # Evidence summary text
        df["evidence_summary"] = df.apply(self._build_evidence_summary, axis=1)

        logger.info(
            "Ranked %d candidates. Top candidate: %s (score=%.3f)",
            len(df),
            df.iloc[0]["gene"],
            df.iloc[0]["composite_score"],
        )
        return df

    @staticmethod
    def _build_evidence_summary(row: pd.Series) -> str:
        """Build a short human-readable evidence string for a candidate."""
        parts = []
        if row.get("cardiac_expression", 0) >= 0.7:
            parts.append("high cardiac expr")
        if row.get("secretome_score", 0) >= 0.7:
            parts.append("confirmed secreted")
        if row.get("plasma_detection", 0) >= 0.7:
            parts.append("plasma-detected")
        if row.get("receptor_coverage", 0) >= 0.5:
            parts.append("multi-organ receptor")
        return "; ".join(parts) if parts else "low evidence"

    def rank_lr_axes(
        self,
        lr_pairs: pd.DataFrame,
        ranked_biomarkers: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rank ligand-receptor axes by composite signal strength.

        Parameters
        ----------
        lr_pairs : pd.DataFrame
            Output of CrossOrganMapper.
        ranked_biomarkers : pd.DataFrame, optional
            Output of ``rank()`` — used to import composite_score per ligand.

        Returns
        -------
        pd.DataFrame
            Columns: ligand, receptor, target_tissue, lr_score,
                     ligand_rank_score, axis_score, axis_rank.
        """
        if lr_pairs.empty:
            return pd.DataFrame()

        df = lr_pairs.copy()

        if ranked_biomarkers is not None and not ranked_biomarkers.empty:
            score_map = ranked_biomarkers.set_index("gene")["composite_score"].to_dict()
            df["ligand_rank_score"] = df["ligand"].map(score_map).fillna(0.5)
        else:
            df["ligand_rank_score"] = 0.5

        df["axis_score"] = (0.6 * df["score"] + 0.4 * df["ligand_rank_score"]).round(4)
        df = df.sort_values("axis_score", ascending=False).reset_index(drop=True)
        df.insert(0, "axis_rank", df.index + 1)

        logger.info("Ranked %d LR axes", len(df))
        return df

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(
        self,
        ranked_biomarkers: pd.DataFrame,
        ranked_lr_axes: Optional[pd.DataFrame] = None,
    ) -> dict[str, Path]:
        """
        Save ranked tables to CSV.

        Returns
        -------
        dict[str, Path]
            Keys: "biomarkers", "lr_axes".
        """
        paths: dict[str, Path] = {}

        bm_path = self._tables_dir / "ranked_biomarkers.csv"
        ranked_biomarkers.to_csv(bm_path, index=False)
        logger.info("Saved ranked biomarkers (%d rows) to %s", len(ranked_biomarkers), bm_path)
        paths["biomarkers"] = bm_path

        if ranked_lr_axes is not None and not ranked_lr_axes.empty:
            lr_path = self._tables_dir / "ranked_lr_axes.csv"
            ranked_lr_axes.to_csv(lr_path, index=False)
            logger.info("Saved ranked LR axes (%d rows) to %s", len(ranked_lr_axes), lr_path)
            paths["lr_axes"] = lr_path

        return paths

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, stem: str) -> list[Path]:
        """Save figure in all configured formats and return paths."""
        saved: list[Path] = []
        for fmt in self.figure_formats:
            out = self._figures_dir / f"{stem}.{fmt}"
            fig.savefig(out, dpi=_FIGURE_DPI, bbox_inches="tight")
            saved.append(out)
            logger.debug("Saved figure: %s", out)
        return saved

    def plot_lollipop(self, ranked_df: pd.DataFrame) -> list[Path]:
        """
        Lollipop chart of top N ranked biomarkers by composite score.

        Parameters
        ----------
        ranked_df : pd.DataFrame
            Output of ``rank()``.

        Returns
        -------
        list[Path]
            Paths to saved figure files.
        """
        top = ranked_df.head(self.top_n).copy()
        top = top.sort_values("composite_score")  # ascending for horizontal plot

        sns.set_style(_FIGURE_STYLE)
        fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.38)))

        # Stems
        ax.hlines(
            y=top["gene"],
            xmin=0,
            xmax=top["composite_score"],
            color="#AAAAAA",
            linewidth=1.5,
            zorder=1,
        )
        # Dots coloured by cardiac_expression
        scatter = ax.scatter(
            x=top["composite_score"],
            y=top["gene"],
            c=top.get("cardiac_expression", 0.5),
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            s=90,
            zorder=2,
            edgecolors="black",
            linewidths=0.4,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label("Cardiac Expression Score", fontsize=9)

        ax.set_xlabel("Composite Ranking Score", fontsize=11)
        ax.set_ylabel("Biomarker Candidate", fontsize=11)
        ax.set_title(f"Top {len(top)} Ranked Cardiac Biomarker Candidates", fontsize=13, pad=14)
        ax.set_xlim(0, 1.05)
        ax.tick_params(labelsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()

        paths = self._save_figure(fig, "ranked_biomarkers_lollipop")
        plt.close(fig)
        return paths

    def plot_forest(self, ranked_df: pd.DataFrame) -> list[Path]:
        """
        Forest-plot style chart showing individual score components
        for each top-N candidate.

        Parameters
        ----------
        ranked_df : pd.DataFrame
            Output of ``rank()``.

        Returns
        -------
        list[Path]
        """
        top = ranked_df.head(self.top_n).copy()
        components = ["cardiac_expression", "secretome_score", "plasma_detection", "receptor_coverage"]
        colors = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261"]
        labels = ["Cardiac Expr.", "Secretome", "Plasma Detect.", "Receptor Cov."]

        n = len(top)
        y_positions = np.arange(n)
        width = 0.18
        offsets = np.linspace(-(len(components) - 1) / 2 * width, (len(components) - 1) / 2 * width, len(components))

        sns.set_style(_FIGURE_STYLE)
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.42)))

        for i, (col, color, label, offset) in enumerate(zip(components, colors, labels, offsets)):
            vals = top[col].values if col in top.columns else np.full(n, 0.5)
            ax.barh(
                y_positions + offset,
                vals,
                height=width,
                color=color,
                alpha=0.85,
                label=label,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(top["gene"].tolist(), fontsize=9)
        ax.set_xlabel("Score (0–1)", fontsize=11)
        ax.set_title(f"Evidence Components — Top {n} Biomarker Candidates", fontsize=13, pad=14)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, 1.1)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
        sns.despine(ax=ax)
        fig.tight_layout()

        paths = self._save_figure(fig, "ranked_biomarkers_forest")
        plt.close(fig)
        return paths

    def plot(self, ranked_df: pd.DataFrame) -> dict[str, list[Path]]:
        """
        Generate all standard ranking plots.

        Returns
        -------
        dict[str, list[Path]]
            Keys: "lollipop", "forest".
        """
        return {
            "lollipop": self.plot_lollipop(ranked_df),
            "forest": self.plot_forest(ranked_df),
        }
