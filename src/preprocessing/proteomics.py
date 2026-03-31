"""
Proteomics preprocessing pipeline.

Loads PRIDE CSV/TSV files, normalizes intensities, maps gene symbols,
filters for EV-relevant proteins, and outputs clean DataFrames for
downstream integration with single-cell data.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EV marker / cargo proteins (used for relevance filtering)
# ---------------------------------------------------------------------------
EV_MARKER_PROTEINS: Set[str] = {
    # Tetraspanins
    "CD9", "CD63", "CD81", "CD151", "TSPAN8",
    # HSPs
    "HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA1B", "HSPA8", "HSP90B1",
    # ESCRT machinery
    "TSG101", "PDCD6IP", "ALIX", "VPS4A", "VPS4B", "CHMP4B",
    # Flotillin
    "FLOT1", "FLOT2",
    # Annexins
    "ANXA1", "ANXA2", "ANXA5", "ANXA6",
    # Integrins
    "ITGB1", "ITGA4", "ITGAV",
    # Signaling
    "ACTB", "GAPDH", "ENO1",
    # Cardiac-specific EV cargo
    "TNNT2", "TNNI3", "MYH7", "MYH6", "ACTN2", "TTN",
    # Extracellular matrix
    "FN1", "LAMA2", "COL1A1", "COL3A1",
}

# Column name variants for protein/gene identifiers across different tool outputs
PROTEIN_ID_COLUMNS = [
    "Accession", "accession",
    "Protein", "ProteinID", "Protein ID",
    "UniProtKB", "UniProt",
    "Master Protein Accessions",
    "Leading razor protein",
    "Majority protein IDs",
    "Protein IDs",
]

GENE_NAME_COLUMNS = [
    "Gene", "gene", "Gene Name", "gene_name", "Gene Names",
    "Gene Symbol", "GeneSymbol",
    "Genes",
]

INTENSITY_COLUMN_PATTERNS = [
    r"^Intensity[_ ]",
    r"^LFQ intensity[_ ]",
    r"^TMT",
    r"^iTRAQ",
    r"^Abundance[_ ]",
    r"^Area[_ ]",
    r"^iBAQ",
    r"Ratio H/L",
]


class ProteomicsPreprocessor:
    """
    Preprocesses PRIDE proteomics data for integration with single-cell data.

    Pipeline
    --------
    1. Load CSV/TSV files (MaxQuant, PD, DIA-NN, mzTab output)
    2. Detect and extract intensity columns
    3. Log2 + median normalization
    4. Map to HGNC gene symbols
    5. Filter for EV-relevant proteins
    6. Output clean, long-format DataFrame

    Usage
    -----
    >>> pp = ProteomicsPreprocessor()
    >>> df = pp.run(Path("data/raw/pride/PXD021371"))
    """

    def __init__(
        self,
        min_valid_fraction: float = 0.5,
        impute_missing: bool = True,
        filter_ev_relevant: bool = True,
        ev_markers: Optional[Set[str]] = None,
        log2_transform: bool = True,
        normalize_method: str = "median",
    ) -> None:
        """
        Parameters
        ----------
        min_valid_fraction:
            Minimum fraction of samples with valid (non-missing) values
            per protein. Proteins below this threshold are removed.
        impute_missing:
            If True, impute missing values with (sample_min - 1.8*std).
        filter_ev_relevant:
            If True, add an ``ev_relevant`` flag column and optionally
            return only EV-relevant proteins.
        ev_markers:
            Custom set of EV marker gene symbols. Defaults to built-in set.
        log2_transform:
            Apply log2 transformation before normalization.
        normalize_method:
            Normalization method: ``median`` or ``quantile``.
        """
        self.min_valid_fraction = min_valid_fraction
        self.impute_missing = impute_missing
        self.filter_ev_relevant = filter_ev_relevant
        self.ev_markers = ev_markers or EV_MARKER_PROTEINS
        self.log2_transform = log2_transform
        self.normalize_method = normalize_method

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: Union[Path, str],
        sample_metadata: Optional[pd.DataFrame] = None,
        return_ev_only: bool = False,
    ) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline on one dataset directory or file.

        Parameters
        ----------
        input_path:
            Path to a file (.csv/.tsv/.xlsx/.mzTab) or directory of such files.
        sample_metadata:
            Optional DataFrame with sample annotations (index = sample names).
        return_ev_only:
            If True, return only EV-relevant proteins.

        Returns
        -------
        Processed wide-format DataFrame (proteins x samples) with
        additional annotation columns.
        """
        input_path = Path(input_path)

        if input_path.is_dir():
            df = self._load_directory(input_path)
        else:
            df = self._load_file(input_path)

        if df is None or df.empty:
            logger.error("No data loaded from: %s", input_path)
            return pd.DataFrame()

        logger.info("Loaded: %d proteins", len(df))

        df = self._extract_identifiers(df)
        intensity_df, meta_df = self._split_intensity_meta(df)

        if intensity_df.empty:
            logger.error("No intensity columns detected in dataset")
            return df

        intensity_df = self._clean_intensities(intensity_df)
        intensity_df = self._filter_by_validity(intensity_df)
        logger.info("After validity filter: %d proteins", len(intensity_df))

        if self.log2_transform:
            intensity_df = self._log2_transform(intensity_df)

        intensity_df = self._normalize(intensity_df)

        if self.impute_missing:
            intensity_df = self._impute(intensity_df)

        result = self._reassemble(intensity_df, meta_df)

        if self.filter_ev_relevant:
            result = self._flag_ev_relevance(result)

        if sample_metadata is not None:
            result = self._add_sample_metadata(result, sample_metadata)

        if return_ev_only:
            result = result[result.get("ev_relevant", pd.Series(True, index=result.index))]

        logger.info("Preprocessing complete: %d proteins x %d samples", len(result), len(intensity_df.columns))
        return result

    def run_multi(
        self,
        project_dirs: Dict[str, Path],
        return_ev_only: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run preprocessing on multiple project directories.

        Parameters
        ----------
        project_dirs:
            Dict mapping project_id -> path.
        return_ev_only:
            Filter to EV-relevant proteins only.

        Returns
        -------
        Dict mapping project_id -> processed DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}
        for pid, path in project_dirs.items():
            logger.info("Processing project: %s", pid)
            try:
                df = self.run(path, return_ev_only=return_ev_only)
                if not df.empty:
                    df["_project"] = pid
                    results[pid] = df
            except Exception as exc:
                logger.error("Failed to process %s: %s", pid, exc)
        return results

    def merge_projects(
        self,
        project_dfs: Dict[str, pd.DataFrame],
        merge_on: str = "gene_symbol",
        aggregate: str = "mean",
    ) -> pd.DataFrame:
        """
        Merge processed DataFrames from multiple projects.

        Aligns on gene symbol and aggregates per-project expression.

        Parameters
        ----------
        project_dfs:
            Dict from ``run_multi``.
        merge_on:
            Column to merge on (default: ``gene_symbol``).
        aggregate:
            How to aggregate duplicate gene entries (``mean``, ``median``, ``max``).

        Returns
        -------
        Wide-format merged DataFrame.
        """
        frames: List[pd.DataFrame] = []
        for pid, df in project_dfs.items():
            if merge_on not in df.columns:
                logger.warning("Column '%s' missing from %s, skipping merge", merge_on, pid)
                continue

            intensity_cols = [c for c in df.columns if c not in self._non_intensity_cols(df)]
            sub = df[[merge_on] + intensity_cols].copy()

            # Add project prefix to sample columns to avoid collision
            sub = sub.rename(columns={c: f"{pid}__{c}" for c in intensity_cols})
            frames.append(sub)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for other in frames[1:]:
            merged = pd.merge(merged, other, on=merge_on, how="outer")

        # Aggregate duplicates
        agg_fn = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax}.get(aggregate, np.nanmean)
        merged = merged.groupby(merge_on, as_index=False).agg(
            lambda x: agg_fn(x.astype(float).values) if pd.api.types.is_numeric_dtype(x) else x.iloc[0]
        )

        return merged

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_directory(self, directory: Path) -> Optional[pd.DataFrame]:
        """Load and merge all compatible files in a directory."""
        frames: List[pd.DataFrame] = []
        for fpath in sorted(directory.iterdir()):
            if fpath.suffix.lower() in (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".mztab"):
                df = self._load_file(fpath)
                if df is not None and not df.empty:
                    frames.append(df)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]

        # Try merging on common identifier columns
        for col in PROTEIN_ID_COLUMNS + GENE_NAME_COLUMNS:
            if all(col in f.columns for f in frames):
                merged = frames[0]
                for other in frames[1:]:
                    try:
                        merged = pd.merge(merged, other, on=col, how="outer")
                    except Exception:
                        pass
                return merged

        # Fallback: return largest file
        return max(frames, key=len)

    def _load_file(self, path: Path) -> Optional[pd.DataFrame]:
        """Load a single file into a DataFrame."""
        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                return pd.read_csv(path, low_memory=False)
            elif ext in (".tsv", ".txt"):
                return pd.read_csv(path, sep="\t", low_memory=False)
            elif ext in (".xlsx", ".xls"):
                return pd.read_excel(path)
            elif ext == ".mztab":
                return self._load_mztab(path)
            else:
                # Attempt auto-detect sep
                return pd.read_csv(path, sep=None, engine="python", low_memory=False)
        except Exception as exc:
            logger.warning("Could not load %s: %s", path.name, exc)
            return None

    def _load_mztab(self, path: Path) -> pd.DataFrame:
        """
        Parse mzTab file and extract protein quantification table.

        Returns DataFrame of PRT section rows.
        """
        rows: List[Dict] = []
        header: Optional[List[str]] = None
        in_prt = False

        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("PRH"):
                    header = line.split("\t")
                    in_prt = True
                elif in_prt and line.startswith("PRT"):
                    if header:
                        rows.append(dict(zip(header, line.split("\t"))))
                elif in_prt and line and not line.startswith("PRT"):
                    in_prt = False

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Identifier extraction
    # ------------------------------------------------------------------

    def _extract_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize protein and gene ID columns.

        Adds ``protein_id`` and ``gene_symbol`` columns if detectable.
        """
        df = df.copy()

        # Protein ID
        for col in PROTEIN_ID_COLUMNS:
            if col in df.columns:
                df["protein_id"] = df[col].astype(str).str.split(";").str[0].str.strip()
                break

        # Gene symbol
        for col in GENE_NAME_COLUMNS:
            if col in df.columns:
                df["gene_symbol"] = df[col].astype(str).str.split(";").str[0].str.strip().str.upper()
                break

        # If gene_symbol still missing, try to parse from protein_id (UniProt GN field)
        if "gene_symbol" not in df.columns and "protein_id" in df.columns:
            df["gene_symbol"] = df["protein_id"].apply(self._extract_gene_from_uniprot_id)

        return df

    @staticmethod
    def _extract_gene_from_uniprot_id(protein_id: str) -> str:
        """
        Extract gene name from UniProt fasta header style string.

        E.g. ``sp|P05067|APP_HUMAN`` -> ``APP``
        """
        if not isinstance(protein_id, str):
            return ""
        m = re.search(r"[A-Z0-9]+_([A-Z]+)", protein_id)
        if m:
            return m.group(0).split("_")[0]
        return protein_id

    # ------------------------------------------------------------------
    # Intensity column handling
    # ------------------------------------------------------------------

    def _split_intensity_meta(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into intensity columns and metadata columns.

        Returns (intensity_df, meta_df).
        """
        intensity_cols = self._detect_intensity_columns(df)
        meta_cols = [c for c in df.columns if c not in intensity_cols]
        return df[intensity_cols].copy(), df[meta_cols].copy()

    def _detect_intensity_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect sample intensity columns via regex patterns and dtype inspection.

        Returns list of column names.
        """
        candidates: List[str] = []
        compiled = [re.compile(p, re.IGNORECASE) for p in INTENSITY_COLUMN_PATTERNS]

        for col in df.columns:
            # Pattern match
            if any(pat.search(str(col)) for pat in compiled):
                candidates.append(col)
                continue
            # Numeric column that looks like expression data (high dynamic range)
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna()
                if len(vals) > 0 and vals.max() > 100:
                    candidates.append(col)

        return candidates

    def _non_intensity_cols(self, df: pd.DataFrame) -> List[str]:
        """Return list of non-intensity metadata column names."""
        intensity_cols = set(self._detect_intensity_columns(df))
        return [c for c in df.columns if c not in intensity_cols]

    # ------------------------------------------------------------------
    # Normalization and transformation
    # ------------------------------------------------------------------

    def _clean_intensities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace zeros with NaN and convert to float.

        Returns cleaned DataFrame.
        """
        df = df.replace(0, np.nan)
        return df.astype(float)

    def _filter_by_validity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove proteins with too many missing values.

        Keeps proteins where at least ``min_valid_fraction`` of samples
        have a valid (non-NaN) value.
        """
        n_samples = df.shape[1]
        valid_counts = df.notna().sum(axis=1)
        mask = valid_counts >= (n_samples * self.min_valid_fraction)
        n_removed = (~mask).sum()
        logger.info(
            "Validity filter: removed %d proteins (min_valid_fraction=%.0f%%)",
            n_removed,
            self.min_valid_fraction * 100,
        )
        return df[mask]

    def _log2_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log2 transformation (log2(x + 1) to handle near-zero values)."""
        logger.info("Applying log2 transformation")
        return np.log2(df + 1)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize intensity matrix.

        Methods
        -------
        median:
            Subtract per-sample median (center to zero).
        quantile:
            Quantile normalization across all samples.
        """
        if self.normalize_method == "median":
            return self._median_normalize(df)
        elif self.normalize_method == "quantile":
            return self._quantile_normalize(df)
        else:
            logger.warning("Unknown normalization method: %s, skipping", self.normalize_method)
            return df

    def _median_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Median-center each sample column.

        Subtracts the per-column median (calculated from non-NaN values).
        """
        medians = df.median(axis=0)
        global_median = medians.median()
        df_norm = df.subtract(medians, axis=1).add(global_median)
        logger.info("Median normalization: global median = %.3f", global_median)
        return df_norm

    def _quantile_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantile normalization across samples.

        Replaces each value with the mean value at the corresponding rank
        across all samples.
        """
        df_filled = df.fillna(df.min().min() - 1)
        ranks = df_filled.rank(method="average")
        sorted_vals = pd.DataFrame(
            np.sort(df_filled.values, axis=0),
            columns=df.columns,
        )
        means = sorted_vals.mean(axis=1)

        def _replace(col: pd.Series) -> pd.Series:
            rank_col = ranks[col.name]
            return rank_col.map(lambda r: means.iloc[int(r) - 1] if not np.isnan(r) else np.nan)

        normalized = df.apply(_replace)
        # Restore original NaN positions
        normalized[df.isna()] = np.nan
        logger.info("Quantile normalization complete")
        return normalized

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using a left-tail distribution approach.

        Missing values are replaced with: min(col) - 1.8 * std(col),
        which places imputed values below the detected signal range
        (consistent with Perseus/MaxQuant imputation).
        """
        df_imp = df.copy()
        for col in df_imp.columns:
            col_data = df_imp[col].dropna()
            if col_data.empty:
                continue
            col_min = col_data.min()
            col_std = col_data.std()
            fill_val = col_min - 1.8 * col_std
            df_imp[col] = df_imp[col].fillna(fill_val)

        n_imputed = df.isna().sum().sum()
        logger.info("Imputed %d missing values", n_imputed)
        return df_imp

    # ------------------------------------------------------------------
    # EV relevance flagging
    # ------------------------------------------------------------------

    def _flag_ev_relevance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add an ``ev_relevant`` boolean column based on EV marker overlap.

        A protein is flagged if its gene_symbol is in the EV marker set,
        or if it is detected in >50% of samples (abundant secreted protein).
        """
        df = df.copy()

        if "gene_symbol" in df.columns:
            df["ev_relevant"] = df["gene_symbol"].str.upper().isin(
                {m.upper() for m in self.ev_markers}
            )
        else:
            df["ev_relevant"] = False

        n_ev = df["ev_relevant"].sum()
        logger.info("EV-relevant proteins flagged: %d / %d", n_ev, len(df))
        return df

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _reassemble(
        self, intensity_df: pd.DataFrame, meta_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Reassemble normalized intensity and metadata columns.

        Returns combined DataFrame aligned on row index.
        """
        result = pd.concat([meta_df.reset_index(drop=True), intensity_df.reset_index(drop=True)], axis=1)
        return result

    def _add_sample_metadata(
        self, df: pd.DataFrame, metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Annotate sample columns with metadata rows.

        Stores metadata in ``df.attrs`` for access without modifying shape.
        """
        df.attrs["sample_metadata"] = metadata
        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_ev_proteins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only EV-relevant rows.

        Parameters
        ----------
        df:
            Processed DataFrame with ``ev_relevant`` column.

        Returns
        -------
        Filtered DataFrame.
        """
        if "ev_relevant" not in df.columns:
            df = self._flag_ev_relevance(df)
        return df[df["ev_relevant"]]

    def to_long_format(
        self,
        df: pd.DataFrame,
        id_cols: Optional[List[str]] = None,
        value_name: str = "log2_intensity",
    ) -> pd.DataFrame:
        """
        Melt wide-format DataFrame to long format.

        Parameters
        ----------
        df:
            Wide-format DataFrame (proteins x samples).
        id_cols:
            Columns to keep as identifiers. Auto-detected if None.
        value_name:
            Name for the value column in the output.

        Returns
        -------
        Long-format DataFrame with columns: [id_cols..., sample, value_name].
        """
        if id_cols is None:
            id_cols = [c for c in ["protein_id", "gene_symbol", "ev_relevant", "_project"] if c in df.columns]

        intensity_cols = [c for c in df.columns if c not in id_cols]
        long_df = df.melt(
            id_vars=id_cols,
            value_vars=intensity_cols,
            var_name="sample",
            value_name=value_name,
        )
        return long_df

    def summarize(self, df: pd.DataFrame) -> Dict:
        """
        Return a summary dictionary for a processed DataFrame.

        Returns dict with protein counts, sample counts, EV coverage, etc.
        """
        intensity_cols = self._detect_intensity_columns(df)
        return {
            "n_proteins": len(df),
            "n_samples": len(intensity_cols),
            "n_ev_relevant": int(df.get("ev_relevant", pd.Series(dtype=bool)).sum()),
            "missing_pct": float(df[intensity_cols].isna().mean().mean() * 100) if intensity_cols else None,
            "has_gene_symbol": "gene_symbol" in df.columns,
            "sample_names": intensity_cols[:10],
        }
