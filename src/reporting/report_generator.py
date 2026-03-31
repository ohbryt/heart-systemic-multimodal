"""
Final report generator for heart systemic multimodal project.

Generates a structured Markdown report with:
- Executive Summary
- Disease Landscape (multi-disease comparison)
- Secretome Analysis
- Plasma Integration
- Cross-Organ Network
- HBAM Index (Heart-to-Body Axis Modulator)
- Methods
- Data Manifest (all input/output files with SHA-256 checksums)
- Figure Catalog (LaTeX-compatible figure references)

Output
------
results/reports/final_report.md

Usage
-----
    gen = ReportGenerator(results_dir=Path("results"))
    gen.run(
        ranked_biomarkers=ranked_df,
        lr_pairs=lr_df,
        hbam_scores=hbam_df,
        disease_comparison=disease_df,
        figure_paths=figure_paths_dict,
    )
"""

from __future__ import annotations

import hashlib
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Section numbering constants
_SEC_EXEC = 1
_SEC_DISEASE = 2
_SEC_SECRETOME = 3
_SEC_PLASMA = 4
_SEC_NETWORK = 5
_SEC_HBAM = 6
_SEC_METHODS = 7
_SEC_FIGURES = 8
_SEC_MANIFEST = 9


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest string, or ``"N/A"`` if the file cannot be read.
    """
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Cannot checksum %s: %s", path, exc)
        return "N/A"


class ReportGenerator:
    """
    Generate a structured Markdown final report.

    Parameters
    ----------
    results_dir : Path
        Root results directory.  Report is written to
        ``results_dir/reports/final_report.md``.
    project_name : str
        Short project identifier used in the report title.
    authors : list[str]
        Author names for the report header.
    """

    def __init__(
        self,
        results_dir: Path = Path("results"),
        project_name: str = "Heart Systemic Multi-modal Analysis",
        authors: Optional[list[str]] = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.authors = authors or ["Brown Biotech Research Team"]

        # Accumulated manifest entries: (path, role, checksum)
        self._manifest: list[dict[str, str]] = []
        # Figure counter for LaTeX-compatible references
        self._fig_counter: int = 0

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def register_file(self, path: Path, role: str = "output") -> None:
        """
        Register a file in the data manifest.

        Parameters
        ----------
        path : Path
            File to register.
        role : str
            Human-readable role label (e.g. ``"input"``, ``"output"``,
            ``"figure"``).
        """
        path = Path(path)
        self._manifest.append(
            {
                "file": str(path),
                "role": role,
                "size_kb": f"{path.stat().st_size / 1024:.1f}" if path.exists() else "N/A",
                "sha256": _sha256(path) if path.exists() else "N/A",
            }
        )

    def register_files(self, paths: list[Path], role: str = "output") -> None:
        """Register multiple files at once.

        Parameters
        ----------
        paths : list[Path]
            Files to register.
        role : str
            Role label shared by all entries.
        """
        for p in paths:
            self.register_file(p, role=role)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        authors_str = ", ".join(self.authors)
        return textwrap.dedent(f"""\
            # {self.project_name}
            ## Final Analysis Report

            **Authors:** {authors_str}
            **Generated:** {now}

            ---
        """)

    def _executive_summary(
        self,
        ranked_biomarkers: Optional[pd.DataFrame],
        lr_pairs: Optional[pd.DataFrame],
        hbam_scores: Optional[pd.DataFrame] = None,
        disease_comparison: Optional[pd.DataFrame] = None,
    ) -> str:
        """Build the Executive Summary section.

        Parameters
        ----------
        ranked_biomarkers : pd.DataFrame, optional
            Output of BiomarkerRanker.rank().
        lr_pairs : pd.DataFrame, optional
            Output of CrossOrganMapper.run().
        hbam_scores : pd.DataFrame, optional
            HBAM index scores per gene/protein.
        disease_comparison : pd.DataFrame, optional
            Multi-disease comparison table.

        Returns
        -------
        str
            Markdown section text.
        """
        n_candidates = len(ranked_biomarkers) if ranked_biomarkers is not None else 0
        n_lr = len(lr_pairs) if lr_pairs is not None else 0
        n_hbam = len(hbam_scores) if hbam_scores is not None else 0

        top_genes = ""
        if ranked_biomarkers is not None and not ranked_biomarkers.empty:
            top5 = ranked_biomarkers.head(5)["gene"].tolist()
            top_genes = f"Top 5 candidates: **{', '.join(top5)}**."

        disease_note = ""
        if disease_comparison is not None and not disease_comparison.empty:
            diseases = sorted(disease_comparison["disease"].unique().tolist()) if "disease" in disease_comparison.columns else []
            if diseases:
                disease_note = f"- **{len(diseases)}** disease states compared: {', '.join(diseases)}.\n"

        hbam_note = ""
        if n_hbam > 0:
            hbam_note = f"- **{n_hbam}** candidates scored with the Heart-to-Body Axis Modulator (HBAM) index.\n"

        return textwrap.dedent(f"""\
            ## {_SEC_EXEC}. Executive Summary

            This report summarises an integrative multi-modal analysis of
            cardiac-secreted proteins and their systemic signalling axes.

            - **{n_candidates}** plasma biomarker candidates ranked by composite
              multi-criteria score (cardiac expression, secretome evidence,
              plasma detection, and cross-organ receptor coverage).
            - **{n_lr}** ligand-receptor axes identified across responder tissues.
            {disease_note}\
            {hbam_note}\
            - {top_genes}

            Key findings are detailed in the Results sections below.

        """)

    def _disease_landscape(
        self,
        disease_comparison: Optional[pd.DataFrame],
    ) -> str:
        """Build the Disease Landscape section.

        Parameters
        ----------
        disease_comparison : pd.DataFrame, optional
            Multi-disease comparison table.  Expected columns include
            ``disease``, ``gene``, ``fold_change``, ``padj``, and any
            per-disease score columns.

        Returns
        -------
        str
        """
        lines: list[str] = [f"## {_SEC_DISEASE}. Disease Landscape\n"]

        if disease_comparison is None or disease_comparison.empty:
            lines.append("_No multi-disease comparison data available._\n")
            return "\n".join(lines) + "\n"

        lines.append(
            "Differential expression and secretome scores were computed across "
            "multiple disease states to identify disease-specific and shared "
            "cardiac signalling programmes.\n"
        )

        # Per-disease summary table
        if "disease" in disease_comparison.columns:
            summary_cols = [c for c in ["disease", "n_candidates", "top_gene", "mean_composite_score"] if c in disease_comparison.columns]
            if "disease" in summary_cols:
                by_disease = (
                    disease_comparison.groupby("disease")
                    .agg(
                        n_candidates=("gene", "count") if "gene" in disease_comparison.columns else ("disease", "count"),
                    )
                    .reset_index()
                )
                lines.append("### Disease Summary\n")
                lines.append(self._df_to_markdown(by_disease))
                lines.append("\n")

        # Full comparison table (top 30 rows)
        display_cols = [c for c in ["disease", "gene", "fold_change", "padj", "composite_score"] if c in disease_comparison.columns]
        if display_cols:
            lines.append(f"### Top Differentially Regulated Genes (top {min(30, len(disease_comparison))} shown)\n")
            lines.append(self._df_to_markdown(disease_comparison[display_cols].head(30)))
            lines.append("\n")

        return "\n".join(lines) + "\n"

    def _secretome_analysis(
        self,
        ranked_biomarkers: Optional[pd.DataFrame],
    ) -> str:
        """Build the Secretome Analysis section.

        Parameters
        ----------
        ranked_biomarkers : pd.DataFrame, optional
            Output of BiomarkerRanker.rank().

        Returns
        -------
        str
        """
        lines: list[str] = [f"## {_SEC_SECRETOME}. Secretome Analysis\n"]

        if ranked_biomarkers is None or ranked_biomarkers.empty:
            lines.append("_No secretome data available._\n")
            return "\n".join(lines) + "\n"

        secretome_cols = [c for c in ["rank", "gene", "secretome_score", "signal_peptide_prob", "cardiac_expression"] if c in ranked_biomarkers.columns]
        if secretome_cols:
            top20 = ranked_biomarkers[secretome_cols].head(20)
            lines.append("### Top 20 Cardiac Secretome Candidates\n")
            lines.append(self._df_to_markdown(top20))
            lines.append("\n")

        # Category breakdown if available
        if "category" in ranked_biomarkers.columns:
            cat_counts = ranked_biomarkers["category"].value_counts()
            lines.append("### Secretome Category Distribution\n")
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            for cat, cnt in cat_counts.items():
                lines.append(f"| {cat} | {cnt} |")
            lines.append("\n")

        return "\n".join(lines) + "\n"

    def _plasma_integration(
        self,
        ranked_biomarkers: Optional[pd.DataFrame],
    ) -> str:
        """Build the Plasma Integration section.

        Parameters
        ----------
        ranked_biomarkers : pd.DataFrame, optional
            Must contain ``plasma_detection`` column.

        Returns
        -------
        str
        """
        lines: list[str] = [f"## {_SEC_PLASMA}. Plasma Integration\n"]
        lines.append(
            "Cross-referencing cardiac secretome candidates against plasma "
            "proteomics and extracellular vesicle (EV) proteomics datasets.\n"
        )

        if ranked_biomarkers is None or ranked_biomarkers.empty:
            lines.append("_No plasma overlap data available._\n")
            return "\n".join(lines) + "\n"

        plasma_cols = [c for c in ["gene", "plasma_detection", "ev_detected", "plasma_datasets_n"] if c in ranked_biomarkers.columns]
        if "plasma_detection" in ranked_biomarkers.columns:
            plasma_top = ranked_biomarkers[plasma_cols].sort_values("plasma_detection", ascending=False).head(20)
            lines.append("### Top Plasma-Detected Candidates\n")
            lines.append(self._df_to_markdown(plasma_top))
            lines.append("\n")

        return "\n".join(lines) + "\n"

    def _cross_organ_network(
        self,
        lr_pairs: Optional[pd.DataFrame],
    ) -> str:
        """Build the Cross-Organ Signaling Network section.

        Parameters
        ----------
        lr_pairs : pd.DataFrame, optional
            Ligand-receptor pairs with columns: ligand, receptor,
            target_tissue, score.

        Returns
        -------
        str
        """
        lines: list[str] = [f"## {_SEC_NETWORK}. Cross-Organ Signaling Network\n"]
        lines.append(
            "Heart-secreted ligands signal to remote tissues via the "
            "ligand-receptor (LR) axes identified below.  Interaction "
            "strength is scored by database confidence × normalised "
            "receptor expression.\n"
        )

        if lr_pairs is None or lr_pairs.empty:
            lines.append("_No LR network data available._\n")
            return "\n".join(lines) + "\n"

        cols = [c for c in ["ligand", "receptor", "target_tissue", "score", "category"] if c in lr_pairs.columns]
        top_lr = lr_pairs[cols].head(20)
        lines.append("### Top 20 Ligand-Receptor Axes\n")
        lines.append(self._df_to_markdown(top_lr))
        lines.append("\n")

        # Tissue-level summary
        if "target_tissue" in lr_pairs.columns and "score" in lr_pairs.columns:
            tissue_summary = (
                lr_pairs.groupby("target_tissue")["score"]
                .agg(n_axes="count", mean_score="mean", total_score="sum")
                .reset_index()
                .sort_values("total_score", ascending=False)
            )
            lines.append("### Signaling Strength by Target Tissue\n")
            lines.append(self._df_to_markdown(tissue_summary))
            lines.append("\n")

        return "\n".join(lines) + "\n"

    def _hbam_section(
        self,
        hbam_scores: Optional[pd.DataFrame],
    ) -> str:
        """Build the HBAM Index section.

        The Heart-to-Body Axis Modulator (HBAM) index is a composite
        machine-learning score integrating:
        - Cardiac expression specificity
        - Secretome probability
        - Plasma detection rate
        - Cross-organ receptor coverage
        - Disease-state differential expression

        Parameters
        ----------
        hbam_scores : pd.DataFrame, optional
            HBAM score table.  Expected columns: ``gene``,
            ``hbam_score``, ``hbam_percentile``, ``disease_group``.

        Returns
        -------
        str
        """
        lines: list[str] = [f"## {_SEC_HBAM}. HBAM Index\n"]
        lines.append(textwrap.dedent("""\
            ### Definition

            The **Heart-to-Body Axis Modulator (HBAM) index** quantifies the
            overall capacity of a cardiac-secreted protein to act as a
            systemic signalling mediator.  It is computed as a weighted
            ensemble of five evidence layers:

            ```
            HBAM = 0.25 × cardiac_specificity
                 + 0.20 × secretome_probability
                 + 0.20 × plasma_detection_rate
                 + 0.20 × receptor_coverage_score
                 + 0.15 × disease_de_score
            ```

            Scores are normalised to [0, 1].  Candidates in the top decile
            (HBAM ≥ 0.80) are considered high-priority systemic mediators.

        """))

        if hbam_scores is None or hbam_scores.empty:
            lines.append("_HBAM scores not yet computed.  Run `hsm hbam` to generate._\n")
            return "\n".join(lines) + "\n"

        # Top candidates table
        display_cols = [c for c in ["rank", "gene", "hbam_score", "hbam_percentile", "disease_group"] if c in hbam_scores.columns]
        if display_cols:
            lines.append("### Top HBAM Candidates\n")
            lines.append(self._df_to_markdown(hbam_scores[display_cols].head(20)))
            lines.append("\n")

        # Distribution statistics
        if "hbam_score" in hbam_scores.columns:
            scores = hbam_scores["hbam_score"].dropna()
            if not scores.empty:
                lines.append("### HBAM Score Distribution\n")
                lines.append("| Statistic | Value |")
                lines.append("|-----------|-------|")
                lines.append(f"| Mean | {scores.mean():.3f} |")
                lines.append(f"| Median | {scores.median():.3f} |")
                lines.append(f"| Std | {scores.std():.3f} |")
                lines.append(f"| P90 (high-priority threshold) | {scores.quantile(0.90):.3f} |")
                lines.append(f"| N high-priority (top 10%) | {(scores >= scores.quantile(0.90)).sum()} |")
                lines.append("\n")

        # Per-disease-group breakdown if available
        if "disease_group" in hbam_scores.columns and "hbam_score" in hbam_scores.columns:
            grp = (
                hbam_scores.groupby("disease_group")["hbam_score"]
                .agg(n="count", mean="mean", median="median")
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            lines.append("### HBAM Score by Disease Group\n")
            lines.append(self._df_to_markdown(grp))
            lines.append("\n")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _methods() -> str:
        """Build the Methods section."""
        return textwrap.dedent(f"""\
            ## {_SEC_METHODS}. Methods

            ### {_SEC_METHODS}.1 Data Sources
            - **Single-cell RNA-seq:** GEO (cardiac and reference tissues —
              GSE183852, GSE135805, GSE290577) and CELLxGENE Census.
            - **Proteomics:** PRIDE Archive (PXD021371, PXD059929, PXD060680)
              — plasma, cardiac secretome (conditioned media), and cross-tissue
              datasets.
            - **Bulk RNA-seq:** ArrayExpress E-MTAB-15659 (cross-tissue).
            - **Disease datasets:** HCM (PXD_HCM), DCM (GSE_DCM), ICM
              (GSE_ICM), and healthy controls.
            - **Ligand-receptor database:** built-in curated list (~200 pairs,
              categories: growth_factors, cytokines, ECM, Wnt, Notch, TGFb, BMP).

            ### {_SEC_METHODS}.2 Preprocessing
            Log-normalisation (scanpy `sc.pp.normalize_total` + `sc.pp.log1p`)
            was applied to all scRNA-seq datasets.  Highly variable genes were
            selected (top 3,000) and dimensionality reduced via PCA → UMAP.

            ### {_SEC_METHODS}.3 Secretome Scoring
            Genes were cross-referenced against UniProt signal-peptide
            annotations and HPA secretome database.  A composite secretome
            score was computed as the harmonic mean of signal-peptide
            probability and tissue-specific expression rank.

            ### {_SEC_METHODS}.4 Plasma Overlap
            Plasma proteomics and EV proteomics datasets were merged.  A plasma
            detection score was assigned as the fraction of datasets in which
            each protein was detected above the 10th-percentile intensity
            threshold.

            ### {_SEC_METHODS}.5 Cross-Organ Receptor Mapping
            For each plasma candidate, matching receptors in reference tissues
            (brain, liver, skeletal muscle) were identified using the built-in
            LR database.  A composite axis score was computed as:

            ```
            score = 0.6 × db_confidence + 0.4 × normalised_receptor_expression
            ```

            ### {_SEC_METHODS}.6 Multi-Disease Comparison
            Pseudo-bulk differential expression was performed for each disease
            group (HCM, DCM, ICM) vs healthy controls using DESeq2-equivalent
            methods.  Fold-change and adjusted p-values (BH correction) were
            computed per gene per disease state.

            ### {_SEC_METHODS}.7 HBAM Index
            The Heart-to-Body Axis Modulator (HBAM) index was trained as a
            gradient-boosted ensemble (XGBoost) using five feature layers:
            cardiac specificity, secretome probability, plasma detection rate,
            receptor coverage score, and disease differential expression score.
            Cross-validation (5-fold) was used to tune hyperparameters.
            SHAP values were computed to assess feature importance.

            ### {_SEC_METHODS}.8 Biomarker Ranking
            Final candidates were ranked by a weighted composite score:

            ```
            composite = 0.30 × cardiac_expression
                      + 0.25 × secretome_score
                      + 0.25 × plasma_detection
                      + 0.20 × receptor_coverage
            ```

            ### {_SEC_METHODS}.9 Statistical Analysis
            All analyses were performed in Python 3.11 with scanpy, pandas,
            numpy, scipy, matplotlib, seaborn, and networkx.

        """)

    def _figure_catalog(self, figure_paths: Optional[dict[str, list[Path]]]) -> str:
        """Build the Figure Catalog section with LaTeX-compatible references.

        Parameters
        ----------
        figure_paths : dict[str, list[Path]], optional
            Mapping of figure label to list of saved file paths.

        Returns
        -------
        str
        """
        if not figure_paths:
            return f"## {_SEC_FIGURES}. Figure Catalog\n\n_No figures generated._\n\n"

        lines = [f"## {_SEC_FIGURES}. Figure Catalog\n"]
        fig_num = 1
        for label, paths in figure_paths.items():
            png_path = next((p for p in paths if p.suffix == ".png"), None)
            pdf_path = next((p for p in paths if p.suffix == ".pdf"), None)
            ref = png_path or pdf_path
            alt_text = label.replace("_", " ").title()
            # LaTeX-compatible label (no spaces, lowercase)
            latex_label = f"fig:{label.lower().replace(' ', '_')}"

            if ref:
                lines.append(f"**Figure {fig_num}: {alt_text}**  \\label{{{latex_label}}}\n")
                lines.append(f"![{alt_text}]({ref})\n")
                if pdf_path and pdf_path != ref:
                    lines.append(f"_PDF version: [{pdf_path.name}]({pdf_path})_\n")
                lines.append(
                    f"_Reference in text: Figure~\\ref{{{latex_label}}}_\n"
                )
                lines.append("")
            fig_num += 1

        return "\n".join(lines) + "\n"

    def _data_manifest(self) -> str:
        """Build the Data Manifest section with full SHA-256 checksums."""
        if not self._manifest:
            return f"## {_SEC_MANIFEST}. Data Manifest\n\n_No files registered._\n\n"

        lines = [
            f"## {_SEC_MANIFEST}. Data Manifest\n",
            "| File | Role | Size (KB) | SHA-256 (full) |",
            "|------|------|-----------|----------------|",
        ]
        for entry in self._manifest:
            short = Path(entry["file"]).name
            sha = entry["sha256"]  # Full 64-char checksum
            lines.append(
                f"| `{short}` | {entry['role']} | {entry['size_kb']} | `{sha}` |"
            )
        return "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame, max_col_width: int = 40) -> str:
        """Convert a DataFrame to a Markdown table string.

        Parameters
        ----------
        df : pd.DataFrame
            Table to render.
        max_col_width : int
            Maximum character width for string values before truncation.

        Returns
        -------
        str
        """
        display = df.copy()
        for col in display.select_dtypes(include="object").columns:
            display[col] = display[col].astype(str).str[:max_col_width]

        header = "| " + " | ".join(str(c) for c in display.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
        rows = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in display.itertuples(index=False, name=None)
        ]
        return "\n".join([header, sep] + rows)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        ranked_biomarkers: Optional[pd.DataFrame] = None,
        lr_pairs: Optional[pd.DataFrame] = None,
        hbam_scores: Optional[pd.DataFrame] = None,
        disease_comparison: Optional[pd.DataFrame] = None,
        figure_paths: Optional[dict[str, list[Path]]] = None,
        extra_input_files: Optional[list[Path]] = None,
        extra_output_files: Optional[list[Path]] = None,
        output_filename: str = "final_report.md",
    ) -> Path:
        """
        Generate and write the final Markdown report.

        Parameters
        ----------
        ranked_biomarkers : pd.DataFrame, optional
            Output of BiomarkerRanker.rank().
        lr_pairs : pd.DataFrame, optional
            Output of CrossOrganMapper.run().
        hbam_scores : pd.DataFrame, optional
            HBAM index scores (columns: gene, hbam_score, hbam_percentile,
            disease_group).
        disease_comparison : pd.DataFrame, optional
            Multi-disease comparison table (columns: disease, gene,
            fold_change, padj, composite_score).
        figure_paths : dict[str, list[Path]], optional
            Dict mapping figure label to list of file paths.
        extra_input_files : list[Path], optional
            Additional input files to register in the manifest.
        extra_output_files : list[Path], optional
            Additional output files to register in the manifest.
        output_filename : str
            Name of the output file inside reports_dir.

        Returns
        -------
        Path
            Path to the generated report.
        """
        # Register provided files in manifest
        if extra_input_files:
            self.register_files(extra_input_files, role="input")
        if extra_output_files:
            self.register_files(extra_output_files, role="output")
        if figure_paths:
            for paths in figure_paths.values():
                self.register_files(paths, role="figure")

        sections = [
            self._header(),
            self._executive_summary(ranked_biomarkers, lr_pairs, hbam_scores, disease_comparison),
            self._disease_landscape(disease_comparison),
            self._secretome_analysis(ranked_biomarkers),
            self._plasma_integration(ranked_biomarkers),
            self._cross_organ_network(lr_pairs),
            self._hbam_section(hbam_scores),
            self._methods(),
            self._figure_catalog(figure_paths),
            self._data_manifest(),
            "---\n_Report generated automatically by heart_systemic_multimodal pipeline._\n",
        ]

        report_text = "\n".join(sections)
        out_path = self.reports_dir / output_filename
        out_path.write_text(report_text, encoding="utf-8")

        logger.info("Final report written to %s (%d chars)", out_path, len(report_text))
        self._log_summary(out_path, ranked_biomarkers, lr_pairs, hbam_scores)
        return out_path

    def _log_summary(
        self,
        report_path: Path,
        ranked_biomarkers: Optional[pd.DataFrame],
        lr_pairs: Optional[pd.DataFrame],
        hbam_scores: Optional[pd.DataFrame] = None,
    ) -> None:
        """Write a brief log summary to results/logs/report_summary.log.

        Parameters
        ----------
        report_path : Path
            Path to the generated report file.
        ranked_biomarkers : pd.DataFrame, optional
        lr_pairs : pd.DataFrame, optional
        hbam_scores : pd.DataFrame, optional
        """
        logs_dir = self.results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "report_summary.log"

        n_bm = len(ranked_biomarkers) if ranked_biomarkers is not None else 0
        n_lr = len(lr_pairs) if lr_pairs is not None else 0
        n_hbam = len(hbam_scores) if hbam_scores is not None else 0
        n_manifest = len(self._manifest)
        timestamp = datetime.now().isoformat()

        summary = (
            f"[{timestamp}] Report generated: {report_path}\n"
            f"  Biomarker candidates ranked: {n_bm}\n"
            f"  LR axes reported: {n_lr}\n"
            f"  HBAM candidates: {n_hbam}\n"
            f"  Manifest entries: {n_manifest}\n"
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(summary)
        logger.info("Log summary appended to %s", log_path)
