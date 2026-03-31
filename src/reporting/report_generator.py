"""
Final report generator for heart systemic multimodal project.

Generates a Markdown report with:
- Executive Summary
- Methods
- Results (per module)
- Discussion
- Data Manifest (all input/output files with SHA-256 checksums)
- Figure Catalog (auto-generated)

Output
------
results/reports/final_report.md

Usage
-----
    gen = ReportGenerator(results_dir=Path("results"))
    gen.run(
        ranked_biomarkers=ranked_df,
        lr_pairs=lr_df,
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


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
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
            Human-readable role label (e.g. "input", "output", "figure").
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
        """Register multiple files."""
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
    ) -> str:
        n_candidates = len(ranked_biomarkers) if ranked_biomarkers is not None else 0
        n_lr = len(lr_pairs) if lr_pairs is not None else 0
        top_genes = ""
        if ranked_biomarkers is not None and not ranked_biomarkers.empty:
            top5 = ranked_biomarkers.head(5)["gene"].tolist()
            top_genes = f"Top 5 candidates: **{', '.join(top5)}**."

        return textwrap.dedent(f"""\
            ## 1. Executive Summary

            This report summarises an integrative multi-modal analysis of
            cardiac-secreted proteins and their systemic signalling axes.

            - **{n_candidates}** plasma biomarker candidates ranked by composite
              multi-criteria score (cardiac expression, secretome evidence,
              plasma detection, and cross-organ receptor coverage).
            - **{n_lr}** ligand-receptor axes identified across responder tissues.
            - {top_genes}

            Key findings are detailed in the Results section below.

        """)

    @staticmethod
    def _methods() -> str:
        return textwrap.dedent("""\
            ## 2. Methods

            ### 2.1 Data Sources
            - Single-cell RNA-seq: GEO (cardiac and reference tissues)
            - Plasma proteomics / EV proteomics: PRIDE/BioStudies
            - Ligand-receptor database: built-in curated list (~200 pairs,
              categories: growth_factors, cytokines, ECM, Wnt, Notch, TGFb, BMP)

            ### 2.2 Preprocessing
            Log-normalisation (scanpy `sc.pp.normalize_total` + `sc.pp.log1p`)
            was applied to all scRNA-seq datasets.  Highly variable genes were
            selected (top 3,000) and dimensionality reduced via PCA → UMAP.

            ### 2.3 Secretome Scoring
            Genes were cross-referenced against UniProt signal-peptide annotations
            and HPA secretome database.  A composite secretome score was computed
            as the harmonic mean of signal-peptide probability and tissue-specific
            expression rank.

            ### 2.4 Plasma Overlap
            Plasma proteomics and EV proteomics datasets were merged.  A plasma
            detection score was assigned as the fraction of datasets in which each
            protein was detected above the 10th-percentile intensity threshold.

            ### 2.5 Cross-Organ Receptor Mapping
            For each plasma candidate, matching receptors in reference tissues
            (brain, liver, skeletal muscle) were identified using the built-in
            LR database.  A composite axis score was computed as:

            ```
            score = 0.6 × db_confidence + 0.4 × normalised_receptor_expression
            ```

            ### 2.6 Biomarker Ranking
            Candidates were ranked by a weighted composite score:

            ```
            composite = 0.30 × cardiac_expression
                      + 0.25 × secretome_score
                      + 0.25 × plasma_detection
                      + 0.20 × receptor_coverage
            ```

            ### 2.7 Statistical Analysis
            All analyses were performed in Python 3.11 with scanpy, pandas,
            numpy, scipy, matplotlib, and seaborn.

        """)

    def _results(
        self,
        ranked_biomarkers: Optional[pd.DataFrame],
        lr_pairs: Optional[pd.DataFrame],
    ) -> str:
        sections: list[str] = ["## 3. Results\n"]

        # --- Biomarker ranking ---
        sections.append("### 3.1 Ranked Biomarker Candidates\n")
        if ranked_biomarkers is not None and not ranked_biomarkers.empty:
            top20 = ranked_biomarkers.head(20)[
                ["rank", "gene", "composite_score", "evidence_summary"]
            ].copy()
            sections.append(self._df_to_markdown(top20))
            sections.append("\n")
        else:
            sections.append("_No ranked biomarkers available._\n\n")

        # --- LR axes summary ---
        sections.append("### 3.2 Top Ligand-Receptor Axes\n")
        if lr_pairs is not None and not lr_pairs.empty:
            cols = [c for c in ["ligand", "receptor", "target_tissue", "score"] if c in lr_pairs.columns]
            top_lr = lr_pairs.head(20)[cols].copy()
            sections.append(self._df_to_markdown(top_lr))
            sections.append("\n")
        else:
            sections.append("_No LR axes available._\n\n")

        return "\n".join(sections)

    @staticmethod
    def _discussion() -> str:
        return textwrap.dedent("""\
            ## 4. Discussion

            The multi-criteria ranking approach highlights candidates with
            convergent evidence across expression, secretion, plasma detection,
            and systemic receptor coverage.  High-ranking candidates represent
            priority targets for:

            1. **Biomarker validation** — ELISA / mass spectrometry in independent
               clinical cohorts.
            2. **Mechanistic follow-up** — receptor blockade or ligand
               neutralisation in relevant cell or animal models.
            3. **Therapeutic targeting** — recombinant ligand administration or
               small-molecule modulation of downstream signalling.

            Limitations include the reliance on publicly available reference
            datasets that may not fully capture disease-specific expression states.
            Future iterations should incorporate disease-matched controls and
            longitudinal plasma samples.

        """)

    def _figure_catalog(self, figure_paths: Optional[dict[str, list[Path]]]) -> str:
        if not figure_paths:
            return "## 5. Figure Catalog\n\n_No figures generated._\n\n"

        lines = ["## 5. Figure Catalog\n"]
        fig_num = 1
        for label, paths in figure_paths.items():
            # Pick PNG for inline reference, PDF as alternative
            png_path = next((p for p in paths if p.suffix == ".png"), None)
            pdf_path = next((p for p in paths if p.suffix == ".pdf"), None)
            ref = png_path or pdf_path
            alt_text = label.replace("_", " ").title()
            if ref:
                lines.append(f"**Figure {fig_num}: {alt_text}**\n")
                lines.append(f"![{alt_text}]({ref})\n")
                if pdf_path and pdf_path != ref:
                    lines.append(f"_PDF version: [{pdf_path.name}]({pdf_path})_\n")
                lines.append("")
            fig_num += 1

        return "\n".join(lines) + "\n"

    def _data_manifest(self) -> str:
        if not self._manifest:
            return "## 6. Data Manifest\n\n_No files registered._\n\n"

        lines = [
            "## 6. Data Manifest\n",
            "| File | Role | Size (KB) | SHA-256 |",
            "|------|------|-----------|---------|",
        ]
        for entry in self._manifest:
            short = Path(entry["file"]).name
            sha = entry["sha256"][:16] + "…" if len(entry["sha256"]) > 16 else entry["sha256"]
            lines.append(
                f"| `{short}` | {entry['role']} | {entry['size_kb']} | `{sha}` |"
            )
        return "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame, max_col_width: int = 40) -> str:
        """Convert a DataFrame to a Markdown table string."""
        # Truncate long string values
        display = df.copy()
        for col in display.select_dtypes(include="object").columns:
            display[col] = display[col].astype(str).str[:max_col_width]

        header = "| " + " | ".join(display.columns) + " |"
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
        figure_paths : dict[str, list[Path]], optional
            Dict mapping figure label → list of file paths.
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
            self._executive_summary(ranked_biomarkers, lr_pairs),
            self._methods(),
            self._results(ranked_biomarkers, lr_pairs),
            self._discussion(),
            self._figure_catalog(figure_paths),
            self._data_manifest(),
            "---\n_Report generated automatically by heart_systemic_multimodal pipeline._\n",
        ]

        report_text = "\n".join(sections)
        out_path = self.reports_dir / output_filename
        out_path.write_text(report_text, encoding="utf-8")

        logger.info("Final report written to %s (%d chars)", out_path, len(report_text))
        self._log_summary(out_path, ranked_biomarkers, lr_pairs)
        return out_path

    def _log_summary(
        self,
        report_path: Path,
        ranked_biomarkers: Optional[pd.DataFrame],
        lr_pairs: Optional[pd.DataFrame],
    ) -> None:
        """Write a brief log summary to results/logs/report_summary.log."""
        logs_dir = self.results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "report_summary.log"

        n_bm = len(ranked_biomarkers) if ranked_biomarkers is not None else 0
        n_lr = len(lr_pairs) if lr_pairs is not None else 0
        n_manifest = len(self._manifest)
        timestamp = datetime.now().isoformat()

        summary = (
            f"[{timestamp}] Report generated: {report_path}\n"
            f"  Biomarker candidates ranked: {n_bm}\n"
            f"  LR axes reported: {n_lr}\n"
            f"  Manifest entries: {n_manifest}\n"
        )
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(summary)
        logger.info("Log summary appended to %s", log_path)
