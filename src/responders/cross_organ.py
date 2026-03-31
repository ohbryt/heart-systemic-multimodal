"""
Cross-organ receptor mapping module.

For each overlapping plasma candidate (secreted from heart), identifies matching
receptors expressed in responder tissues (brain, liver, muscle, etc.) using a
CellPhoneDB/CellChat-style ligand-receptor approach.

Output: lr_pairs.csv with columns:
    ligand, receptor, source_tissue, target_tissue, score,
    receptor_coverage, systemic_reach

Additional outputs:
    cross_organ_heatmap.png — ligands x organs heatmap coloured by receptor expression

Usage
-----
    mapper = CrossOrganMapper(
        plasma_candidates=["VEGFA", "TGFB1", "BNP"],
        output_dir=Path("results/tables"),
    )
    mapper.load_reference_datasets(
        brain_path=Path("data/brain_ref.h5ad"),
        liver_path=Path("data/liver_ref.h5ad"),
        muscle_path=Path("data/muscle_ref.h5ad"),
    )
    df = mapper.run()
    mapper.save(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lr_database import LRDatabase

logger = logging.getLogger(__name__)

# Minimum mean expression to consider a receptor "expressed" in a tissue
_DEFAULT_EXPR_THRESHOLD = 0.1
# Weight given to expression level vs database confidence in final score
_EXPR_WEIGHT = 0.4
_CONF_WEIGHT = 0.6


class CrossOrganMapper:
    """
    Map cardiac-secreted plasma candidates to receptors in responder tissues.

    Parameters
    ----------
    plasma_candidates : list[str]
        Gene symbols of plasma-detected ligand candidates (secreted by heart).
    output_dir : Path
        Directory where lr_pairs.csv will be written.
    lr_db : LRDatabase, optional
        Pre-instantiated LRDatabase. If None, a default one is created.
    expr_threshold : float
        Minimum mean expression (normalised counts) to call a receptor expressed.
    extra_lr_path : Path, optional
        Extra LR pairs YAML/CSV to augment the built-in database.
    """

    def __init__(
        self,
        plasma_candidates: list[str],
        output_dir: Path = Path("results/tables"),
        lr_db: Optional[LRDatabase] = None,
        expr_threshold: float = _DEFAULT_EXPR_THRESHOLD,
        extra_lr_path: Optional[Path] = None,
    ) -> None:
        self.plasma_candidates = [g.upper() for g in plasma_candidates]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.expr_threshold = expr_threshold
        self.lr_db = lr_db if lr_db is not None else LRDatabase(extra_path=extra_lr_path)

        # Dict[tissue_name -> mean expression Series (index = gene)]
        self._tissue_expression: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_reference_datasets(
        self,
        brain_path: Optional[Path] = None,
        liver_path: Optional[Path] = None,
        muscle_path: Optional[Path] = None,
        **extra_tissues: Path,
    ) -> None:
        """
        Load optional reference datasets for responder tissues.

        Each path should point to an AnnData (.h5ad) file with log-normalised
        counts in ``adata.X``.  Missing paths are silently skipped.

        Parameters
        ----------
        brain_path : Path, optional
        liver_path : Path, optional
        muscle_path : Path, optional
        **extra_tissues : Path
            Additional tissues as keyword arguments, e.g. ``kidney=Path(...)``.
        """
        named_paths: dict[str, Optional[Path]] = {
            "brain": brain_path,
            "liver": liver_path,
            "muscle": muscle_path,
        }
        named_paths.update({k: v for k, v in extra_tissues.items()})

        for tissue, path in named_paths.items():
            if path is None:
                logger.debug("No reference dataset provided for %s — skipping", tissue)
                continue
            self._tissue_expression[tissue] = self._load_mean_expression(tissue, Path(path))

        if not self._tissue_expression:
            logger.warning(
                "No reference datasets loaded. Running in database-only mode "
                "(scores will be based solely on LR database confidence)."
            )

    @staticmethod
    def _load_mean_expression(tissue: str, path: Path) -> pd.Series:
        """
        Load mean gene expression from an AnnData h5ad file.

        Falls back to an empty Series if anndata is unavailable or the file
        does not exist.
        """
        if not path.exists():
            logger.warning("Reference file not found for %s: %s", tissue, path)
            return pd.Series(dtype=float)
        try:
            import anndata  # noqa: PLC0415

            adata = anndata.read_h5ad(path)
            # Compute per-gene mean across all cells
            import scipy.sparse as sp  # noqa: PLC0415

            X = adata.X
            if sp.issparse(X):
                mean_expr = np.asarray(X.mean(axis=0)).flatten()
            else:
                mean_expr = np.asarray(X).mean(axis=0)
            series = pd.Series(mean_expr, index=adata.var_names, dtype=float)
            logger.info(
                "Loaded %s reference: %d genes, %d cells", tissue, adata.n_vars, adata.n_obs
            )
            return series
        except ImportError:
            logger.error("anndata not installed — cannot load reference dataset for %s", tissue)
            return pd.Series(dtype=float)
        except Exception as exc:
            logger.error("Failed to load reference for %s from %s: %s", tissue, path, exc)
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Core mapping logic
    # ------------------------------------------------------------------

    def _score_lr_pair(
        self,
        ligand: str,
        receptor: str,
        tissue: str,
        db_confidence: float,
    ) -> float:
        """
        Compute a composite LR pair score.

        Score = conf_weight * db_confidence + expr_weight * expr_score
        where expr_score is the normalised receptor expression in the target
        tissue (clipped to [0, 1]).  If no expression data is available,
        score falls back to db_confidence.
        """
        expr_series = self._tissue_expression.get(tissue)
        if expr_series is None or expr_series.empty:
            return float(db_confidence)

        raw_expr = float(expr_series.get(receptor, 0.0))
        # Normalise: assume max relevant expression ~5 (log-normalised counts)
        expr_score = min(raw_expr / 5.0, 1.0)

        if raw_expr < self.expr_threshold:
            return 0.0  # receptor not expressed — discard

        score = _CONF_WEIGHT * db_confidence + _EXPR_WEIGHT * expr_score
        return round(float(score), 4)

    def _map_single_candidate(
        self,
        ligand: str,
        source_tissue: str = "heart",
    ) -> list[dict[str, object]]:
        """
        Find all LR pairs for one ligand across all loaded target tissues.

        Returns a list of row dicts for the output DataFrame.
        """
        receptors = self.lr_db.get_receptors(ligand)
        if not receptors:
            logger.debug("No receptors found in DB for ligand %s", ligand)
            return []

        rows: list[dict[str, object]] = []

        # If no tissue expression data, use "unknown" as target tissue
        target_tissues = list(self._tissue_expression.keys()) or ["unknown"]

        for target_tissue in target_tissues:
            for receptor in receptors:
                db_conf = self.lr_db.get_pair_confidence(ligand, receptor)
                score = self._score_lr_pair(ligand, receptor, target_tissue, db_conf)
                if score <= 0.0:
                    continue
                rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "source_tissue": source_tissue,
                        "target_tissue": target_tissue,
                        "db_confidence": round(db_conf, 4),
                        "score": score,
                    }
                )

        return rows

    def run(self, source_tissue: str = "heart") -> pd.DataFrame:
        """
        Run cross-organ receptor mapping for all plasma candidates.

        Parameters
        ----------
        source_tissue : str
            Label for the secreting tissue (default: "heart").

        Returns
        -------
        pd.DataFrame
            Columns: ligand, receptor, source_tissue, target_tissue,
                     db_confidence, score.
            Sorted by score descending.
        """
        logger.info(
            "Running cross-organ mapping for %d plasma candidates", len(self.plasma_candidates)
        )
        all_rows: list[dict[str, object]] = []

        for ligand in self.plasma_candidates:
            rows = self._map_single_candidate(ligand, source_tissue=source_tissue)
            logger.debug("  %s → %d receptor hits", ligand, len(rows))
            all_rows.extend(rows)

        if not all_rows:
            logger.warning("No LR pairs found. Check plasma candidates and reference data.")
            return pd.DataFrame(
                columns=[
                    "ligand",
                    "receptor",
                    "source_tissue",
                    "target_tissue",
                    "db_confidence",
                    "score",
                ]
            )

        df = (
            pd.DataFrame(all_rows)
            .drop_duplicates(subset=["ligand", "receptor", "source_tissue", "target_tissue"])
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            "Cross-organ mapping complete: %d LR pairs across %d target tissues",
            len(df),
            df["target_tissue"].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str = "lr_pairs.csv") -> Path:
        """
        Write the LR pairs DataFrame to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``run()``.
        filename : str
            Output filename (written inside ``output_dir``).

        Returns
        -------
        Path
            Full path of the written file.
        """
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False)
        logger.info("Saved LR pairs (%d rows) to %s", len(df), out_path)
        return out_path

    # ------------------------------------------------------------------
    # Receptor coverage and systemic reach
    # ------------------------------------------------------------------

    def receptor_coverage_score(self, ligand: str) -> dict[str, float]:
        """
        Compute the fraction of loaded target tissues that express at least
        one matched receptor above ``expr_threshold`` for a given ligand.

        Parameters
        ----------
        ligand : str
            Gene symbol of the cardiac-secreted ligand.

        Returns
        -------
        dict[str, float]
            Mapping ``receptor -> coverage_fraction`` where coverage_fraction
            is the proportion of target tissues in which that receptor is
            expressed at or above ``expr_threshold``.
        """
        receptors = self.lr_db.get_receptors(ligand.upper())
        if not receptors:
            return {}

        target_tissues = list(self._tissue_expression.keys())
        n_tissues = len(target_tissues)
        if n_tissues == 0:
            # No expression data — coverage is undefined; return 1.0 (database-only)
            return {r: 1.0 for r in receptors}

        coverage: dict[str, float] = {}
        for receptor in receptors:
            expressed_in = sum(
                1
                for tissue in target_tissues
                if float(
                    self._tissue_expression[tissue].get(receptor, 0.0)
                ) >= self.expr_threshold
            )
            coverage[receptor] = round(expressed_in / n_tissues, 4)
        return coverage

    def systemic_reach(self, ligand: str) -> int:
        """
        Count the number of distinct organs/tissues in which at least one
        matched receptor for ``ligand`` is expressed above threshold.

        Parameters
        ----------
        ligand : str

        Returns
        -------
        int
            Number of organs with confirmed receptor expression.
        """
        receptors = self.lr_db.get_receptors(ligand.upper())
        if not receptors:
            return 0

        target_tissues = list(self._tissue_expression.keys())
        if not target_tissues:
            return 0

        organs_with_receptor: set[str] = set()
        for tissue in target_tissues:
            expr = self._tissue_expression[tissue]
            for receptor in receptors:
                if float(expr.get(receptor, 0.0)) >= self.expr_threshold:
                    organs_with_receptor.add(tissue)
                    break  # one receptor sufficient to count the organ
        return len(organs_with_receptor)

    def plot_cross_organ_heatmap(
        self,
        df: pd.DataFrame,
        save: bool = True,
        filename: str = "cross_organ_heatmap.png",
    ) -> plt.Figure:
        """
        Plot a heatmap of ligands x target_tissues coloured by mean receptor
        expression in each tissue.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``run()`` with at least columns
            ``ligand``, ``target_tissue``, ``score``.
        save : bool
            If True, saves the figure to ``output_dir / filename``.
        filename : str

        Returns
        -------
        matplotlib.figure.Figure
        """
        if df.empty:
            logger.warning("Empty LR pairs DataFrame — skipping heatmap")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        pivot = (
            df.groupby(["ligand", "target_tissue"])["score"]
            .mean()
            .unstack(fill_value=0.0)
        )

        fig, ax = plt.subplots(
            figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot.index) * 0.5))
        )
        try:
            import seaborn as sns  # noqa: PLC0415
            sns.heatmap(
                pivot,
                cmap="YlOrRd",
                linewidths=0.4,
                annot=pivot.shape[0] <= 20,
                fmt=".2f",
                ax=ax,
                cbar_kws={"label": "Mean LR Score"},
            )
        except ImportError:
            im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            fig.colorbar(im, ax=ax, label="Mean LR Score")

        ax.set_title("Cross-Organ Receptor Heatmap (Ligands × Organs)")
        ax.set_xlabel("Target Tissue")
        ax.set_ylabel("Ligand")
        plt.tight_layout()

        if save:
            out_path = self.output_dir / filename
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info("Cross-organ heatmap saved to %s", out_path)
        plt.close(fig)
        return fig

    def _enrich_with_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ``receptor_coverage`` and ``systemic_reach`` columns to an LR
        pairs DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``run()``.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with two additional columns.
        """
        if df.empty:
            df["receptor_coverage"] = pd.Series(dtype=float)
            df["systemic_reach"] = pd.Series(dtype=int)
            return df

        # Pre-compute per-ligand coverage dicts and reach values
        ligands = df["ligand"].unique()
        coverage_map: dict[str, dict[str, float]] = {
            lig: self.receptor_coverage_score(lig) for lig in ligands
        }
        reach_map: dict[str, int] = {
            lig: self.systemic_reach(lig) for lig in ligands
        }

        df = df.copy()
        df["receptor_coverage"] = df.apply(
            lambda row: coverage_map[row["ligand"]].get(row["receptor"], 0.0),
            axis=1,
        )
        df["systemic_reach"] = df["ligand"].map(reach_map)
        return df

    def run_and_save(
        self,
        source_tissue: str = "heart",
        filename: str = "lr_pairs.csv",
    ) -> tuple[pd.DataFrame, Path]:
        """
        Convenience wrapper: run mapping, enrich with coverage metrics, and save.

        The saved CSV includes ``receptor_coverage`` and ``systemic_reach``
        columns in addition to the base LR pair columns.

        Returns
        -------
        tuple[pd.DataFrame, Path]
        """
        df = self.run(source_tissue=source_tissue)
        df = self._enrich_with_coverage(df)
        path = self.save(df, filename=filename)
        return df, path
