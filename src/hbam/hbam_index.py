"""
HBAM — Heart-Based Aging and Multiorgan Index.

Aggregates per-sample features from phenotype scoring, secretome analysis,
plasma overlap detection, and cross-organ receptor mapping into a single
composite index.  Supports training, prediction, and SHAP-based explanation.

Outputs
-------
``hbam_scores.csv`` — per-sample HBAM scores
``hbam_model.pkl``  — serialised sklearn model
``hbam_shap.png``   — SHAP summary bar plot
``hbam_dist.png``   — HBAM score distribution (violin + swarm)

Example
-------
    from src.hbam import HBAMIndex, HBAMFeatureBuilder

    builder = HBAMFeatureBuilder(output_dir=Path("results/tables"))
    features = builder.build(adata, secretome_df, plasma_df, lr_df)

    hbam = HBAMIndex(output_dir=Path("results"))
    score = hbam.compute_hbam(adata, secretome_df, plasma_df, lr_df)
    print(score)   # single float (cohort mean)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from .feature_builder import HBAMFeatureBuilder

logger = logging.getLogger(__name__)

# Default model hyper-parameters
_RF_N_ESTIMATORS: int = 200
_RF_MAX_DEPTH: int = 6
_RF_RANDOM_STATE: int = 42


class HBAMIndex:
    """
    Heart-Based Aging and Multiorgan Index.

    Parameters
    ----------
    output_dir : Path
        Root directory for all output files.
    feature_builder : HBAMFeatureBuilder, optional
        Pre-configured feature builder.  A default one is created if None.
    sample_col : str
        Column in ``adata.obs`` identifying biological samples.
    """

    def __init__(
        self,
        output_dir: Path = Path("results"),
        feature_builder: Optional[HBAMFeatureBuilder] = None,
        sample_col: str = "sample_id",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_col = sample_col
        self._builder = feature_builder or HBAMFeatureBuilder(
            output_dir=self.output_dir / "tables",
            sample_col=sample_col,
        )
        self._model: Optional[object] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_hbam(
        self,
        adata,
        secretome_df: pd.DataFrame,
        plasma_df: pd.DataFrame,
        lr_df: pd.DataFrame,
    ) -> float:
        """
        Compute a cohort-level HBAM index (mean over all samples).

        Parameters
        ----------
        adata : AnnData
            Single-cell dataset with phenotype scores in ``adata.obs``.
        secretome_df : pd.DataFrame
            Secretome ranking table (must have ``gene`` column).
        plasma_df : pd.DataFrame
            Plasma overlap table (must have ``gene`` column).
        lr_df : pd.DataFrame
            LR pairs from CrossOrganMapper.

        Returns
        -------
        float
            Cohort mean HBAM score (mean of normalised feature row-norms).
        """
        features = self._builder.build(adata, secretome_df, plasma_df, lr_df)
        # Simple heuristic composite: L2 row norm of the feature vector
        row_norms = np.linalg.norm(features.values, axis=1)
        hbam_score = float(np.mean(row_norms))
        logger.info("Cohort HBAM index: %.4f", hbam_score)
        return hbam_score

    def train_hbam_model(
        self,
        features_df: pd.DataFrame,
        labels: Union[pd.Series, np.ndarray],
        n_estimators: int = _RF_N_ESTIMATORS,
        max_depth: int = _RF_MAX_DEPTH,
        cv: int = 5,
    ) -> RandomForestRegressor:
        """
        Train a RandomForest regressor to predict HBAM labels.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix (samples x features).  Typically the output of
            ``HBAMFeatureBuilder.build()``.
        labels : array-like
            Continuous target labels (e.g. biological age, disease severity).
        n_estimators : int
        max_depth : int
        cv : int
            Number of cross-validation folds for logging purposes.

        Returns
        -------
        RandomForestRegressor
            Fitted model.  Also stored as ``self._model``.
        """
        X = features_df.values
        y = np.asarray(labels).ravel()

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=_RF_RANDOM_STATE,
            n_jobs=-1,
        )
        cv_scores = cross_val_score(model, X, y, cv=min(cv, len(y)), scoring="r2")
        logger.info(
            "CV R² (k=%d): %.3f ± %.3f", cv, np.mean(cv_scores), np.std(cv_scores)
        )

        model.fit(X, y)
        self._model = model
        logger.info("HBAM model trained (%d trees, max_depth=%d)", n_estimators, max_depth)
        return model

    def predict_hbam(
        self,
        model: object,
        features_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict HBAM scores for new samples.

        Parameters
        ----------
        model
            Fitted sklearn estimator (e.g. from ``train_hbam_model``).
        features_df : pd.DataFrame
            Feature matrix aligned to the training feature set.

        Returns
        -------
        pd.Series
            Predicted HBAM scores indexed by sample_id.
        """
        preds = model.predict(features_df.values)
        result = pd.Series(preds, index=features_df.index, name="hbam_score")
        logger.info("Predicted HBAM scores for %d samples", len(result))
        return result

    def explain_hbam(
        self,
        model: object,
        features_df: pd.DataFrame,
        max_display: int = 15,
        save: bool = True,
    ) -> "np.ndarray":
        """
        Compute SHAP values and optionally save a summary bar plot.

        Parameters
        ----------
        model
            Fitted sklearn tree ensemble (RandomForest or XGBoost).
        features_df : pd.DataFrame
        max_display : int
            Number of top features to display in the SHAP plot.
        save : bool
            If True, saves the plot to ``output_dir/hbam_shap.png``.

        Returns
        -------
        np.ndarray
            SHAP values array of shape (n_samples, n_features).
        """
        try:
            import shap  # noqa: PLC0415
        except ImportError:
            logger.error("shap is not installed. Run: pip install shap")
            return np.zeros((len(features_df), features_df.shape[1]))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df.values)

        # SHAP bar summary plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            shap_values,
            features_df,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        if save:
            out_path = self.output_dir / "hbam_shap.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP plot saved to %s", out_path)
        plt.close(fig)

        return np.asarray(shap_values)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_distribution(
        self,
        scores: pd.Series,
        group_col: Optional[str] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Violin + swarm plot of HBAM score distribution.

        Parameters
        ----------
        scores : pd.Series
            Per-sample HBAM scores.
        group_col : str, optional
            If provided, the series name or a column in a companion DataFrame
            used to colour swarm points by group.
        save : bool
            If True, saves to ``output_dir/hbam_dist.png``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import seaborn as sns  # noqa: PLC0415
        except ImportError:
            logger.warning("seaborn not installed — falling back to simple histogram")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(scores.values, bins=20, color="#2A6EBB", edgecolor="white")
            ax.set_xlabel("HBAM Score")
            ax.set_ylabel("Count")
            ax.set_title("HBAM Score Distribution")
            plt.tight_layout()
            if save:
                out_path = self.output_dir / "hbam_dist.png"
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                logger.info("Distribution plot saved to %s", out_path)
            plt.close(fig)
            return fig

        plot_df = scores.reset_index()
        plot_df.columns = ["sample_id", "hbam_score"]
        plot_df["group"] = plot_df.get(group_col, "all") if group_col else "all"

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(
            data=plot_df, x="group", y="hbam_score",
            palette="Set2", inner=None, ax=ax,
        )
        sns.swarmplot(
            data=plot_df, x="group", y="hbam_score",
            color="0.25", size=4, ax=ax,
        )
        ax.set_xlabel("Group")
        ax.set_ylabel("HBAM Score")
        ax.set_title("HBAM Score Distribution")
        plt.tight_layout()

        if save:
            out_path = self.output_dir / "hbam_dist.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info("Distribution plot saved to %s", out_path)
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_scores(
        self,
        scores: pd.Series,
        filename: str = "hbam_scores.csv",
    ) -> Path:
        """
        Persist per-sample HBAM scores to CSV.

        Parameters
        ----------
        scores : pd.Series
        filename : str

        Returns
        -------
        Path
        """
        out_path = self.output_dir / filename
        scores.to_csv(out_path, header=True)
        logger.info("HBAM scores (%d samples) saved to %s", len(scores), out_path)
        return out_path

    def save_model(
        self,
        model: object,
        filename: str = "hbam_model.pkl",
    ) -> Path:
        """
        Serialise a trained model to disk with pickle.

        Parameters
        ----------
        model
            Any sklearn-compatible fitted estimator.
        filename : str

        Returns
        -------
        Path
        """
        out_path = self.output_dir / filename
        with open(out_path, "wb") as fh:
            pickle.dump(model, fh)
        logger.info("HBAM model saved to %s", out_path)
        return out_path

    @staticmethod
    def load_model(model_path: Path) -> object:
        """
        Load a previously saved model.

        Parameters
        ----------
        model_path : Path

        Returns
        -------
        object
            Deserialised sklearn estimator.
        """
        with open(Path(model_path), "rb") as fh:
            model = pickle.load(fh)
        logger.info("HBAM model loaded from %s", model_path)
        return model
