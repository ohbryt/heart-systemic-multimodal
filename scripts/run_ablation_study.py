#!/usr/bin/env python
"""Ablation study: quantify each modality's contribution to IRIS performance."""

import argparse
import logging
import torch
from src.data.cardiac_loader import load_cardiac_data
from src.data.graph_builder import GraphBuilder
from src.data.cardiac_labels import build_cardiac_labels
from src.models.iris_model import IrisModel
from src.training.trainer import IrisTrainer
from src.training.evaluator import IrisEvaluator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_ablation(
    modality_data, graph, labels, exclude_modality=None, epochs=50, patience=15
):
    """Train with specified modality excluded."""
    if exclude_modality:
        data = {k: v for k, v in modality_data.items() if k != exclude_modality}
    else:
        data = modality_data

    model = IrisModel(modality_features={k: v.shape[1] for k, v in data.items()})
    trainer = IrisTrainer(model=model, epochs=epochs, patience=patience)
    history = trainer.train(data, graph, labels)

    model.eval()
    with torch.no_grad():
        scores = model(data, graph)

    evaluator = IrisEvaluator()
    return evaluator.evaluate(scores, labels)


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on IRIS modalities"
    )
    parser.add_argument("--n-genes", type=int, default=1000, help="Number of genes")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("IRIS Ablation Study")
    logger.info("=" * 60)

    logger.info("Loading data (%d genes)...", args.n_genes)
    data, genes, gene_ids, modality_features = load_cardiac_data(
        max_genes=args.n_genes, seed=42
    )
    builder = GraphBuilder(use_api=False, seed=42)
    graph = builder.build(gene_ids, gene_names=genes)
    labels = build_cardiac_labels(genes, gene_ids, seed=42)

    modality_tensors = {k: v.features for k, v in data.items()}

    n_pos = int(labels.sum().item())
    logger.info(
        "Labels: %d positive out of %d (%.1f%%)",
        n_pos,
        len(labels),
        100 * n_pos / len(labels),
    )

    results = {}

    logger.info("\n=== Baseline (all modalities) ===")
    baseline = run_ablation(
        modality_tensors,
        graph,
        labels,
        exclude_modality=None,
        epochs=args.epochs,
        patience=args.patience,
    )
    results["baseline"] = baseline
    logger.info(
        "AUROC: %.3f, AUPRC: %.3f, nDCG@20: %.3f",
        baseline["auroc"],
        baseline["auprc"],
        baseline["ndcg_20"],
    )

    logger.info("\n=== Leave-One-Out Ablation ===")
    for mod in modality_tensors.keys():
        logger.info("\n--- Without %s ---", mod)
        result = run_ablation(
            modality_tensors,
            graph,
            labels,
            exclude_modality=mod,
            epochs=args.epochs,
            patience=args.patience,
        )
        results[mod] = result
        delta_auroc = result["auroc"] - baseline["auroc"]
        delta_auprc = result["auprc"] - baseline["auprc"]
        logger.info(
            "AUROC: %.3f (Δ%.3f), AUPRC: %.3f (Δ%.3f)",
            result["auroc"],
            delta_auroc,
            result["auprc"],
            delta_auprc,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info("%-20s %10s %10s", "Modality", "AUROC", "AUPRC")
    logger.info("-" * 42)
    logger.info(
        "%-20s %10.3f %10.3f", "Baseline (all)", baseline["auroc"], baseline["auprc"]
    )
    for mod, result in results.items():
        if mod == "baseline":
            continue
        delta_auroc = result["auroc"] - baseline["auroc"]
        delta_auprc = result["auprc"] - baseline["auprc"]
        marker = " **" if abs(delta_auroc) > 0.02 else ""
        logger.info("%-20s %+.3f %+.3f%s", f"-{mod}", delta_auroc, delta_auprc, marker)

    logger.info("\n** indicates >2% AUROC change")


if __name__ == "__main__":
    main()
