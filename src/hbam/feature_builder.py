"""
HBAM Feature Builder.

Constructs the per-sample feature matrix used to train and predict the
Heart-Based Aging and Multiorgan (HBAM) index.

Feature sources
---------------
1. Phenotype scores — mean per-sample across 7 cardiac programs
   (fibrosis, ECM_remodeling, inflammation, oxidative_stress,
    metabolic_shift, cardiomyocyte_stress, angiogenesis)
2. Top secretome gene expression — mean expression of the N highest-ranked
   secretome candidates per sample
3. Plasma detection binary vector — 1 if the gene was detected in plasma,
   0 otherwise
4. LR pair count — number of validated ligand-receptor pairs per sample

All features are normalised with sklearn StandardScaler before output.

Output
------
``features_matrix.csv`` — rows = samples, cols = feature names.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Canonical phenotype score column names expected in ``adata.obs``
_PHENOTYPE_COLS: list[str] = [
    "fibrosis",
    "ECM_remodeling",
    "inflammation",
    "oxidative_stress",
    "metabolic_shift",
    "cardiomyocyte_stress",
    "angiogenesis",
]

# Default number of top secretome genes to include
_DEFAULT_TOP_N_SECRETOME: int = 20


class HBAMFeatureBuilder:
    """
    Build the HBAM feature matrix for a cohort of samples.

    Parameters
    ----------
    output_dir : Path
        Directory where ``features_matrix.csv`` will be saved.
    top_n_secretome : int
        Number of top-ranked secretome candidates to use for expression features.
    sample_col : str
        Column in ``adata.obs`` that identifies biological samples/patients.
    """

    def __init__(
        self,
        output_dir: Path = Path("results/tables"),
        top_n_secretome: int = _DEFAULT_TOP_N_SECRETOME,
        sample_col: str = "sample_id",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_n_secretome = top_n_secretome
        self.sample_col = sample_col
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        adata,
        secretome_df: pd.DataFrame,
        plasma_df: pd.DataFrame,
        lr_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct the full feature matrix.

        Parameters
        ----------
        adata : AnnData
            Single-cell dataset with phenotype scores stored in ``adata.obs``
            and log-normalised counts in ``adata.X``.
        secretome_df : pd.DataFrame
            Secretome ranking table. Must contain a ``gene`` column;
            rows should be ordered by rank (highest confidence first).
        plasma_df : pd.DataFrame
            Plasma overlap table. Must contain a ``gene`` column; each row
            is a gene detected in plasma.
        lr_df : pd.DataFrame
            LR pairs table from CrossOrganMapper. Must contain ``ligand``
            and ``source_tissue`` columns.

        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per sample and a ``sample_id`` index.
            Values are StandardScaler-normalised.
        """
        samples = self._get_samples(adata)
        logger.info("Building HBAM features for %d samples", len(samples))

        pheno_feats = self._extract_phenotype_features(adata, samples)
        sec_feats = self._extract_secretome_features(adata, secretome_df, samples)
        plasma_feats = self._extract_plasma_features(plasma_df, samples)
        lr_feats = self._extract_lr_features(lr_df, samples)

        features = pd.concat([pheno_feats, sec_feats, plasma_feats, lr_feats], axis=1)
        features = self._fill_missing(features)
        features_norm = self._normalise(features)

        logger.info(
            "Feature matrix: %d samples x %d features",
            features_norm.shape[0],
            features_norm.shape[1],
        )
        return features_norm

    def save(self, features: pd.DataFrame, filename: str = "features_matrix.csv") -> Path:
        """
        Write feature matrix to CSV.

        Parameters
        ----------
        features : pd.DataFrame
        filename : str

        Returns
        -------
        Path
        """
        out_path = self.output_dir / filename
        features.to_csv(out_path)
        logger.info("Saved feature matrix (%d x %d) to %s", *features.shape, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_samples(self, adata) -> list[str]:
        """Return ordered unique sample identifiers from adata.obs."""
        if self.sample_col not in adata.obs.columns:
            logger.warning(
                "Column '%s' not found in adata.obs — treating all cells as one sample",
                self.sample_col,
            )
            return ["all"]
        return list(adata.obs[self.sample_col].unique())

    def _extract_phenotype_features(self, adata, samples: list[str]) -> pd.DataFrame:
        """
        Compute mean phenotype score per sample for each of the 7 programs.

        Missing score columns are filled with 0.
        """
        obs = adata.obs.copy()
        if self.sample_col not in obs.columns:
            obs[self.sample_col] = "all"

        available_cols = [c for c in _PHENOTYPE_COLS if c in obs.columns]
        missing = set(_PHENOTYPE_COLS) - set(available_cols)
        if missing:
            logger.warning("Phenotype score columns missing in adata.obs: %s", missing)
            for col in missing:
                obs[col] = 0.0

        grouped = obs.groupby(self.sample_col)[_PHENOTYPE_COLS].mean()
        grouped.index.name = "sample_id"
        grouped.columns = [f"pheno_{c}" for c in grouped.columns]
        return grouped.reindex(samples)

    def _extract_secretome_features(
        self, adata, secretome_df: pd.DataFrame, samples: list[str]
    ) -> pd.DataFrame:
        """
        Compute mean expression of the top-N secretome genes per sample.

        Returns a single column ``secretome_mean_expr``.
        """
        if "gene" not in secretome_df.columns:
            logger.warning("secretome_df missing 'gene' column — skipping secretome features")
            return pd.DataFrame(
                {"secretome_mean_expr": 0.0}, index=pd.Index(samples, name="sample_id")
            )

        top_genes: list[str] = secretome_df["gene"].iloc[: self.top_n_secretome].tolist()
        # Intersect with genes present in adata
        try:
            available = [g for g in top_genes if g in adata.var_names]
        except Exception:
            available = []

        if not available:
            logger.warning("None of the top secretome genes found in adata.var_names")
            return pd.DataFrame(
                {"secretome_mean_expr": 0.0}, index=pd.Index(samples, name="sample_id")
            )

        obs = adata.obs.copy()
        if self.sample_col not in obs.columns:
            obs[self.sample_col] = "all"

        import scipy.sparse as sp  # noqa: PLC0415

        gene_idx = [list(adata.var_names).index(g) for g in available]
        X = adata.X
        if sp.issparse(X):
            sub = np.asarray(X[:, gene_idx].mean(axis=1)).flatten()
        else:
            sub = np.asarray(X)[:, gene_idx].mean(axis=1)

        obs["_sec_expr"] = sub
        result = obs.groupby(self.sample_col)["_sec_expr"].mean().rename("secretome_mean_expr")
        result.index.name = "sample_id"
        return result.reindex(samples).to_frame()

    def _extract_plasma_features(
        self, plasma_df: pd.DataFrame, samples: list[str]
    ) -> pd.DataFrame:
        """
        Return a single binary feature: fraction of plasma genes detected.

        Since plasma overlap is not per-sample in the standard pipeline, this
        returns a constant column. Extend this method when per-sample plasma
        proteomics is available.
        """
        if plasma_df is None or plasma_df.empty or "gene" not in plasma_df.columns:
            logger.warning("plasma_df missing or has no 'gene' column — returning 0")
            n_detected = 0
        else:
            n_detected = len(plasma_df["gene"].dropna().unique())

        return pd.DataFrame(
            {"plasma_overlap_count": float(n_detected)},
            index=pd.Index(samples, name="sample_id"),
        )

    def _extract_lr_features(
        self, lr_df: pd.DataFrame, samples: list[str]
    ) -> pd.DataFrame:
        """
        Return per-sample LR pair count.

        If lr_df has a ``sample_id`` column the count is per-sample;
        otherwise a global constant is broadcast to all samples.
        """
        if lr_df is None or lr_df.empty:
            logger.warning("lr_df is empty — returning 0 LR count")
            return pd.DataFrame(
                {"lr_pair_count": 0.0},
                index=pd.Index(samples, name="sample_id"),
            )

        if "sample_id" in lr_df.columns:
            counts = (
                lr_df.groupby("sample_id")
                .size()
                .rename("lr_pair_count")
                .reindex(samples, fill_value=0.0)
            )
            counts.index.name = "sample_id"
            return counts.to_frame()

        # Global constant
        return pd.DataFrame(
            {"lr_pair_count": float(len(lr_df))},
            index=pd.Index(samples, name="sample_id"),
        )

    @staticmethod
    def _fill_missing(features: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with column median; remaining NaN with 0."""
        features = features.fillna(features.median(numeric_only=True))
        features = features.fillna(0.0)
        return features

    def _normalise(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply StandardScaler; store fitted scaler for inference reuse."""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features.values)
        self._scaler = scaler
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)
