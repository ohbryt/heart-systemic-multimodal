"""
Heart-Systemic Multi-modal Pipeline — CLI Entry Point.

Usage:
    python main.py [COMMAND] [OPTIONS]
    hsm [COMMAND] [OPTIONS]  (after pip install -e .)

Commands:
    download     Fetch raw datasets from GEO / PRIDE / CELLxGENE.
    preprocess   QC, normalize, and embed all enabled datasets.
    score        Score cells for fibrosis, ECM, inflammation, and other programs.
    secretome    Identify candidate cardiac secretome proteins.
    overlap      Cross-tissue overlap analysis (heart vs liver/muscle/etc.).
    responders   Identify systemic tissue responders to cardiac signals.
    rank         Rank and prioritize secretome candidates.
    hbam         Compute the Heart-to-Body Axis Modulator (HBAM) index.
    train-hbam   Train the ML model backing the HBAM index.
    report       Generate HTML / PDF analysis report.
    run-all      Execute the full pipeline end-to-end (includes hbam step).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.traceback import install as install_rich_traceback

from src.utils.config_loader import load_config
from src.utils.logger import setup_logging, log_section, log_config_summary

# Install Rich as the default traceback renderer for nicer error output.
install_rich_traceback(show_locals=False, width=120)

console = Console(stderr=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared CLI options
# ---------------------------------------------------------------------------

_CONFIG_OPTION = click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=False, path_type=Path),
    help="Path to a user YAML config override file.",
    show_default=True,
)

_VERBOSE_OPTION = click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)

_DRY_RUN_OPTION = click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be done without executing.",
)


def _init(
    config_path: Optional[Path],
    verbose: bool,
    project_root: Optional[Path] = None,
) -> tuple:
    """Bootstrap config and logging for a CLI command.

    Args:
        config_path: Optional user config override path.
        verbose: If True, use DEBUG logging level.
        project_root: Project root for path resolution.

    Returns:
        Tuple of (ConfigLoader, log_file_path).
    """
    cfg = load_config(
        user_config_path=config_path,
        project_root=project_root or Path.cwd(),
    )
    level = "DEBUG" if verbose else cfg.get("execution", "log_level") or "INFO"
    log_dir = cfg.get_path("logs_dir") if "logs_dir" in cfg.config.get("paths", {}) else None
    log_file = setup_logging(level=level, log_dir=log_dir, force=True)
    log_config_summary(cfg.config, logger=logger)
    return cfg, log_file


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.version_option("0.1.0", prog_name="hsm")
def cli() -> None:
    """Heart-Systemic Multi-modal Analysis Pipeline.

    Integrates scRNA-seq, proteomics, and bulk RNA-seq data to identify
    cardiac secretome candidates and cross-tissue systemic responders.

    Run 'hsm COMMAND --help' for command-specific options.
    """


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

@cli.command("download")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--dataset",
    "-d",
    "datasets",
    multiple=True,
    help="Specific dataset name(s) to download. Repeat for multiple. "
         "Default: all enabled datasets.",
)
def cmd_download(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    datasets: tuple[str, ...],
) -> None:
    """Fetch raw datasets from GEO, PRIDE, and CELLxGENE Census.

    Downloads are skipped for datasets whose local_path already contains
    the expected files.  Set --dry-run to preview without downloading.

    Examples:

        hsm download

        hsm download --dataset GSE183852 --dataset GSE135805

        hsm download --config my_overrides.yaml --dry-run
    """
    cfg, _ = _init(config_path, verbose)
    log_section("DOWNLOAD")

    from src.data.downloader import DatasetDownloader

    enabled = cfg.enabled_datasets()
    targets = {k: v for k, v in enabled.items() if not datasets or k in datasets}

    if not targets:
        logger.warning("No matching enabled datasets found. Check your config.")
        sys.exit(0)

    logger.info("Datasets to download: %s", list(targets))

    downloader = DatasetDownloader(config=cfg.config, dry_run=dry_run)
    results = downloader.download_all(targets)

    ok = sum(1 for s in results.values() if s == "ok")
    skipped = sum(1 for s in results.values() if s == "skipped")
    failed = sum(1 for s in results.values() if s == "failed")
    logger.info("Download complete: %d ok, %d skipped, %d failed.", ok, skipped, failed)

    if failed:
        logger.error("Some downloads failed. Check logs for details.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

@cli.command("preprocess")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--dataset",
    "-d",
    "datasets",
    multiple=True,
    help="Specific dataset name(s) to preprocess.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run preprocessing even if cached output exists.",
)
def cmd_preprocess(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    datasets: tuple[str, ...],
    force: bool,
) -> None:
    """QC filter, normalize, embed, and cluster all enabled datasets.

    Steps per dataset:
      1. Load raw data (h5ad / 10x mtx / CSV)
      2. QC filtering (min_cells, min_genes, max_mito)
      3. Normalization + log1p
      4. Highly variable gene selection
      5. PCA -> Harmony batch correction -> UMAP
      6. Leiden clustering
      7. Save processed AnnData to processed_dir

    Examples:

        hsm preprocess

        hsm preprocess --dataset heart_cellxgene --force
    """
    cfg, _ = _init(config_path, verbose)
    log_section("PREPROCESS")

    from src.data.preprocessor import Preprocessor

    enabled = cfg.enabled_datasets()
    targets = {k: v for k, v in enabled.items() if not datasets or k in datasets}

    if not targets:
        logger.warning("No datasets to preprocess.")
        sys.exit(0)

    prep_cfg = cfg.config.get("preprocessing", {})
    preprocessor = Preprocessor(config=cfg.config, force=force, dry_run=dry_run)

    failed: list[str] = []
    for name, spec in targets.items():
        logger.info("Preprocessing: %s", name)
        try:
            preprocessor.run(name, spec)
        except Exception as exc:  # noqa: BLE001
            logger.error("Preprocessing failed for %s: %s", name, exc, exc_info=verbose)
            failed.append(name)

    if failed:
        logger.error("Failed datasets: %s", failed)
        sys.exit(1)

    logger.info("Preprocessing complete.")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

@cli.command("score")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--program",
    "-p",
    "programs",
    multiple=True,
    help="Gene program(s) to score (e.g. fibrosis, ECM). Default: all.",
)
@click.option(
    "--dataset",
    "-d",
    "datasets",
    multiple=True,
    help="Dataset(s) to score. Default: all preprocessed.",
)
def cmd_score(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    programs: tuple[str, ...],
    datasets: tuple[str, ...],
) -> None:
    """Score cells using curated gene programs (fibrosis, ECM, inflammation, etc.).

    Uses scanpy.tl.score_genes() for each program defined in the 'scoring'
    config block.  Scores are added as .obs columns to each processed AnnData
    and the updated objects are saved back to processed_dir.

    Examples:

        hsm score

        hsm score --program fibrosis --program hypertrophy

        hsm score --dataset heart_failure_GSE183852
    """
    cfg, _ = _init(config_path, verbose)
    log_section("SCORE")

    from src.analysis.scorer import GeneSetScorer

    gene_sets = cfg.scoring_gene_sets()
    if programs:
        gene_sets = {k: v for k, v in gene_sets.items() if k in programs}

    if not gene_sets:
        logger.error("No matching gene programs found. Available: %s", list(cfg.scoring_gene_sets()))
        sys.exit(1)

    logger.info("Scoring programs: %s", list(gene_sets))

    enabled = cfg.enabled_datasets()
    targets = {k: v for k, v in enabled.items() if not datasets or k in datasets}

    scorer = GeneSetScorer(config=cfg.config, gene_sets=gene_sets, dry_run=dry_run)

    failed: list[str] = []
    for name in targets:
        try:
            scorer.score_dataset(name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Scoring failed for %s: %s", name, exc, exc_info=verbose)
            failed.append(name)

    if failed:
        sys.exit(1)

    logger.info("Scoring complete.")


# ---------------------------------------------------------------------------
# secretome
# ---------------------------------------------------------------------------

@cli.command("secretome")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--min-expr-fraction",
    default=None,
    type=float,
    help="Override config min_expr_fraction for secreted gene detection.",
)
def cmd_secretome(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    min_expr_fraction: Optional[float],
) -> None:
    """Identify candidate cardiac secretome proteins.

    Integrates:
      - scRNA-seq: highly expressed, predicted secreted genes per cell type
      - Proteomics: proteins detected in cardiac secretome datasets (PXD059929)
      - Signal peptide / secretion prediction (precomputed DB or SignalP)

    Output: results/tables/secretome_candidates.tsv

    Examples:

        hsm secretome

        hsm secretome --min-expr-fraction 0.1
    """
    cfg, _ = _init(config_path, verbose)
    log_section("SECRETOME")

    from src.analysis.secretome import SecretomeAnalyzer

    override_cfg = cfg.config.copy()
    if min_expr_fraction is not None:
        override_cfg.setdefault("analysis", {}).setdefault("secretome", {})[
            "min_expr_fraction"
        ] = min_expr_fraction

    analyzer = SecretomeAnalyzer(config=override_cfg, dry_run=dry_run)
    results = analyzer.run()

    out_path = Path(cfg.get_path("tables_dir")) / "secretome_candidates.tsv"
    if not dry_run:
        results.to_csv(out_path, sep="\t", index=False)
        logger.info("Secretome candidates saved: %s (%d rows)", out_path, len(results))
    else:
        logger.info("[dry-run] Would save %d secretome candidates to %s", len(results), out_path)


# ---------------------------------------------------------------------------
# overlap
# ---------------------------------------------------------------------------

@cli.command("overlap")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--tissue",
    "-t",
    "tissues",
    multiple=True,
    help="Comparison tissue(s) (e.g. liver, skeletal_muscle). Default: config list.",
)
def cmd_overlap(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    tissues: tuple[str, ...],
) -> None:
    """Cross-tissue overlap analysis: heart vs systemic tissues.

    Identifies genes and proteins differentially regulated in heart failure
    that are also altered in liver, skeletal muscle, or other tissues.
    Uses Fisher's exact test and Jaccard index for overlap scoring.

    Output: results/tables/cross_tissue_overlap.tsv

    Examples:

        hsm overlap

        hsm overlap --tissue liver --tissue skeletal_muscle
    """
    cfg, _ = _init(config_path, verbose)
    log_section("OVERLAP")

    from src.analysis.overlap import CrossTissueOverlap

    overlap_cfg = cfg.config.get("analysis", {}).get("overlap", {})
    comparison_tissues = list(tissues) or overlap_cfg.get("comparison_tissues", [])

    if not comparison_tissues:
        logger.error("No comparison tissues specified. Use --tissue or set analysis.overlap.comparison_tissues in config.")
        sys.exit(1)

    logger.info("Comparing heart vs: %s", comparison_tissues)

    analyzer = CrossTissueOverlap(config=cfg.config, dry_run=dry_run)
    results = analyzer.run(comparison_tissues=comparison_tissues)

    out_path = Path(cfg.get_path("tables_dir")) / "cross_tissue_overlap.tsv"
    if not dry_run:
        results.to_csv(out_path, sep="\t", index=False)
        logger.info("Overlap results saved: %s (%d rows)", out_path, len(results))


# ---------------------------------------------------------------------------
# responders
# ---------------------------------------------------------------------------

@cli.command("responders")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--min-correlation",
    default=None,
    type=float,
    help="Minimum correlation coefficient to consider a responder.",
)
def cmd_responders(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    min_correlation: Optional[float],
) -> None:
    """Identify systemic tissue cells that respond to cardiac secreted factors.

    Correlates fibrosis/hypertrophy scores in heart cells with transcriptional
    responses in systemic tissues (liver, muscle, etc.).
    Requires responder datasets to be enabled and preprocessed.

    Output: results/tables/systemic_responders.tsv

    Examples:

        hsm responders

        hsm responders --min-correlation 0.4
    """
    cfg, _ = _init(config_path, verbose)
    log_section("RESPONDERS")

    from src.analysis.responders import ResponderIdentifier

    resp_cfg = cfg.config.get("analysis", {}).get("responders", {})
    if min_correlation is not None:
        resp_cfg["min_correlation"] = min_correlation

    responder_datasets = {
        k: v for k, v in cfg.enabled_datasets().items()
        if k in ("liver_responder", "muscle_responder", "custom_responder")
    }

    if not responder_datasets:
        logger.warning(
            "No responder datasets enabled. "
            "Set liver_responder/muscle_responder/custom_responder enabled: true in config."
        )
        sys.exit(0)

    identifier = ResponderIdentifier(config=cfg.config, dry_run=dry_run)
    results = identifier.run(responder_datasets=responder_datasets)

    out_path = Path(cfg.get_path("tables_dir")) / "systemic_responders.tsv"
    if not dry_run:
        results.to_csv(out_path, sep="\t", index=False)
        logger.info("Responder results saved: %s (%d rows)", out_path, len(results))


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

@cli.command("rank")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--top-n",
    default=None,
    type=int,
    help="Number of top candidates to report per category.",
)
@click.option(
    "--output-format",
    type=click.Choice(["tsv", "csv", "xlsx"], case_sensitive=False),
    default="tsv",
    show_default=True,
    help="Output file format.",
)
@click.option(
    "--disease",
    "-d",
    "diseases",
    multiple=True,
    help="Filter ranking to specific disease group(s) (e.g. HCM, DCM, ICM). Default: all.",
)
def cmd_rank(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    top_n: Optional[int],
    output_format: str,
    diseases: tuple[str, ...],
) -> None:
    """Rank and prioritize secretome candidates using a composite score.

    Aggregates evidence from:
      - scRNA-seq expression scores
      - Proteomics detection scores
      - Ligand-receptor interaction scores
      - Cross-tissue overlap scores
      - Literature evidence scores (optional)

    Weights are defined in config analysis.ranking.weights.

    Output: results/tables/ranked_candidates.{tsv|csv|xlsx}

    Examples:

        hsm rank

        hsm rank --top-n 30 --output-format xlsx
    """
    cfg, _ = _init(config_path, verbose)
    log_section("RANK")

    from src.analysis.ranker import CandidateRanker

    rank_cfg = cfg.config.get("analysis", {}).get("ranking", {})
    if top_n is not None:
        rank_cfg["top_n"] = top_n
    if diseases:
        rank_cfg["disease_filter"] = list(diseases)
    effective_top_n = rank_cfg.get("top_n", 20)

    ranker = CandidateRanker(config=cfg.config, dry_run=dry_run)
    ranked = ranker.run()

    # Apply disease filter post-hoc if column is present
    if diseases and "disease" in ranked.columns:
        ranked = ranked[ranked["disease"].isin(diseases)]
        logger.info("Disease filter applied: %s (%d rows remain)", list(diseases), len(ranked))

    suffix = output_format.lower()
    out_path = Path(cfg.get_path("tables_dir")) / f"ranked_candidates.{suffix}"

    if not dry_run:
        if suffix == "xlsx":
            ranked.to_excel(out_path, index=False)
        elif suffix == "csv":
            ranked.to_csv(out_path, index=False)
        else:
            ranked.to_csv(out_path, sep="\t", index=False)
        logger.info(
            "Ranked candidates saved: %s (top %d of %d)", out_path, effective_top_n, len(ranked)
        )
    else:
        logger.info("[dry-run] Would save %d ranked candidates to %s", len(ranked), out_path)


# ---------------------------------------------------------------------------
# hbam
# ---------------------------------------------------------------------------

@cli.command("hbam")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--disease",
    "-d",
    "diseases",
    multiple=True,
    help="Disease group(s) to include (e.g. HCM, DCM, ICM). Default: all.",
)
@click.option(
    "--output-format",
    type=click.Choice(["tsv", "csv", "xlsx"], case_sensitive=False),
    default="tsv",
    show_default=True,
    help="Output file format.",
)
def cmd_hbam(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    diseases: tuple[str, ...],
    output_format: str,
) -> None:
    """Compute the Heart-to-Body Axis Modulator (HBAM) index.

    The HBAM index integrates five evidence layers into a single composite
    score per candidate gene/protein:
      - Cardiac expression specificity
      - Secretome probability
      - Plasma detection rate
      - Cross-organ receptor coverage score
      - Disease differential expression score

    Requires ranked_candidates.tsv and secretome_candidates.tsv from
    previous pipeline steps.

    Output: results/tables/hbam_scores.{tsv|csv|xlsx}

    Examples:

        hsm hbam

        hsm hbam --disease HCM --disease DCM

        hsm hbam --output-format xlsx
    """
    cfg, _ = _init(config_path, verbose)
    log_section("HBAM")

    from src.scoring.hbam_index import HBAMScorer

    hbam_cfg = cfg.config.get("analysis", {}).get("hbam", {})
    if diseases:
        hbam_cfg["disease_filter"] = list(diseases)

    scorer = HBAMScorer(config=hbam_cfg, dry_run=dry_run)
    results = scorer.run()

    suffix = output_format.lower()
    out_path = Path(cfg.get_path("tables_dir")) / f"hbam_scores.{suffix}"

    if not dry_run:
        if suffix == "xlsx":
            results.to_excel(out_path, index=False)
        elif suffix == "csv":
            results.to_csv(out_path, index=False)
        else:
            results.to_csv(out_path, sep="\t", index=False)
        logger.info("HBAM scores saved: %s (%d rows)", out_path, len(results))
    else:
        logger.info("[dry-run] Would save %d HBAM scores to %s", len(results), out_path)


# ---------------------------------------------------------------------------
# train-hbam
# ---------------------------------------------------------------------------

@cli.command("train-hbam")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--disease",
    "-d",
    "diseases",
    multiple=True,
    help="Disease group(s) to include in training (e.g. HCM, DCM, ICM). Default: all.",
)
@click.option(
    "--cv-folds",
    default=5,
    type=int,
    show_default=True,
    help="Number of cross-validation folds for hyperparameter tuning.",
)
@click.option(
    "--model-output",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to save the trained model (default: results/models/hbam_model.pkl).",
)
def cmd_train_hbam(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    diseases: tuple[str, ...],
    cv_folds: int,
    model_output: Optional[Path],
) -> None:
    """Train the ML model backing the HBAM index.

    Trains a gradient-boosted ensemble (XGBoost) using the five HBAM
    feature layers.  Cross-validation is used to tune hyperparameters
    and SHAP values are computed for interpretability.

    Requires labelled training data in results/tables/hbam_training_data.tsv
    (or path specified in config analysis.hbam.training_data_path).

    Output:
      - results/models/hbam_model.pkl   (trained model)
      - results/tables/hbam_shap.tsv    (SHAP feature importances)
      - results/figures/shap_feature_importance.{pdf,png}

    Examples:

        hsm train-hbam

        hsm train-hbam --cv-folds 10 --disease HCM --disease DCM

        hsm train-hbam --model-output /path/to/model.pkl
    """
    cfg, _ = _init(config_path, verbose)
    log_section("TRAIN-HBAM")

    from src.scoring.hbam_index import HBAMTrainer

    hbam_cfg = cfg.config.get("analysis", {}).get("hbam", {})
    if diseases:
        hbam_cfg["disease_filter"] = list(diseases)
    hbam_cfg["cv_folds"] = cv_folds

    default_model_out = Path(cfg.get_path("results_dir")) / "models" / "hbam_model.pkl"
    model_path = model_output or default_model_out

    trainer = HBAMTrainer(config=hbam_cfg, dry_run=dry_run)
    result = trainer.train(model_output_path=model_path)

    if not dry_run:
        logger.info(
            "HBAM model trained: CV score=%.4f, saved to %s",
            result.get("cv_score", float("nan")),
            model_path,
        )
    else:
        logger.info("[dry-run] Would train HBAM model and save to %s", model_path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@cli.command("report")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["html", "pdf", "notebook"], case_sensitive=False),
    default="html",
    show_default=True,
    help="Output report format.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    default=False,
    help="Open the generated report in the default browser.",
)
def cmd_report(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    report_format: str,
    open_browser: bool,
) -> None:
    """Generate an analysis summary report (HTML / PDF / Jupyter Notebook).

    Collects all results from previous pipeline steps and renders a
    structured report with figures, tables, and interpretation text.

    Output: results/report/analysis_report.{html|pdf|ipynb}

    Examples:

        hsm report

        hsm report --format pdf

        hsm report --format html --open
    """
    cfg, _ = _init(config_path, verbose)
    log_section("REPORT")

    from src.visualization.reporter import ReportGenerator

    reporter = ReportGenerator(config=cfg.config, dry_run=dry_run)
    out_path = reporter.generate(fmt=report_format)

    if out_path and open_browser and not dry_run:
        import webbrowser
        webbrowser.open(f"file://{out_path.resolve()}")
        logger.info("Opened report in browser: %s", out_path)


# ---------------------------------------------------------------------------
# run-all
# ---------------------------------------------------------------------------

@cli.command("run-all")
@_CONFIG_OPTION
@_VERBOSE_OPTION
@_DRY_RUN_OPTION
@click.option(
    "--skip",
    "skip_steps",
    multiple=True,
    type=click.Choice(
        ["download", "preprocess", "score", "secretome", "overlap", "responders", "rank", "hbam", "report"],
        case_sensitive=False,
    ),
    help="Step(s) to skip. Repeat for multiple.",
)
@click.option(
    "--start-from",
    "start_from",
    default=None,
    type=click.Choice(
        ["download", "preprocess", "score", "secretome", "overlap", "responders", "rank", "hbam", "report"],
        case_sensitive=False,
    ),
    help="Resume pipeline from this step (skips earlier steps).",
)
def cmd_run_all(
    config_path: Optional[Path],
    verbose: bool,
    dry_run: bool,
    skip_steps: tuple[str, ...],
    start_from: Optional[str],
) -> None:
    """Execute the complete analysis pipeline end-to-end.

    Runs all steps in order:
      download -> preprocess -> score -> secretome -> overlap ->
      responders -> rank -> hbam -> report

    Use --skip to omit individual steps, or --start-from to resume
    from a checkpoint.

    Examples:

        hsm run-all

        hsm run-all --skip download --skip report

        hsm run-all --start-from score --config my_config.yaml

        hsm run-all --skip hbam --dry-run
    """
    cfg, _ = _init(config_path, verbose)
    log_section("RUN-ALL")

    all_steps = [
        "download",
        "preprocess",
        "score",
        "secretome",
        "overlap",
        "responders",
        "rank",
        "hbam",
        "report",
    ]

    # Determine which steps to run.
    active_steps = all_steps[:]
    if start_from:
        start_idx = all_steps.index(start_from)
        active_steps = all_steps[start_idx:]
    active_steps = [s for s in active_steps if s not in skip_steps]

    logger.info("Pipeline steps to execute: %s", active_steps)
    if dry_run:
        logger.info("[dry-run] No changes will be made.")

    # Map step names to Click commands so we can invoke them programmatically.
    step_map = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "score": cmd_score,
        "secretome": cmd_secretome,
        "overlap": cmd_overlap,
        "responders": cmd_responders,
        "rank": cmd_rank,
        "hbam": cmd_hbam,
        "report": cmd_report,
    }

    failed_steps: list[str] = []
    for step in active_steps:
        log_section(f"STEP: {step}")
        try:
            ctx = click.get_current_context()
            cmd = step_map[step]
            # Build minimal args; each sub-command reloads config internally.
            extra: list[str] = []
            if verbose:
                extra.append("--verbose")
            if dry_run:
                extra.append("--dry-run")
            if config_path:
                extra.extend(["--config", str(config_path)])
            ctx.invoke(cmd, config_path=config_path, verbose=verbose, dry_run=dry_run)
        except SystemExit as exc:
            if exc.code != 0:
                logger.error("Step '%s' exited with code %s. Stopping pipeline.", step, exc.code)
                failed_steps.append(step)
                break
        except Exception as exc:  # noqa: BLE001
            logger.error("Step '%s' failed: %s", step, exc, exc_info=True)
            failed_steps.append(step)
            break

    if failed_steps:
        logger.error("Pipeline failed at step(s): %s", failed_steps)
        sys.exit(1)

    log_section("PIPELINE COMPLETE")
    logger.info("All steps finished successfully.")
    logger.info("Results are in: %s", cfg.get_path("results_dir"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
