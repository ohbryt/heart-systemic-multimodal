#!/usr/bin/env python3
"""Run the IRIS multimodal deep model pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

from src.data.graph_builder import GraphBuilder
from src.data.label_builder import build_labels
from src.data.synthetic_generator import MODALITY_FEATURES, SyntheticDataGenerator
from src.models.iris_model import IrisModel
from src.training.evaluator import IrisEvaluator
from src.training.trainer import IrisTrainer
from src.utils.io_utils import ensure_parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="IRIS multimodal deep model")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--real", action="store_true", help="Use real data via A3 ingestion")
    parser.add_argument("--n-genes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/reports")
    args = parser.parse_args()

    use_real = args.real and not args.synthetic
    logger.info("IRIS pipeline starting (mode=%s, n_genes=%d)",
                "real" if use_real else "synthetic", args.n_genes)

    gene_names: list[str] | None = None

    if use_real:
        from src.ingestion.orchestrator import IngestionOrchestrator
        orch = IngestionOrchestrator(
            registry_path="data/registry/dataset_registry.csv",
            output_dir="data",
            n_genes_fallback=args.n_genes,
            seed=args.seed,
        )
        data = orch.run()
        gene_ids = data["bulk_rna"].gene_ids
        gene_names = data["bulk_rna"].gene_names
        modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES if mod in data}
        args.n_genes = len(gene_ids)
    elif args.synthetic:
        gen = SyntheticDataGenerator(n_genes=args.n_genes, seed=args.seed)
        data = gen.generate()
        gene_ids = data["bulk_rna"].gene_ids
        gene_names = data["bulk_rna"].gene_names
        modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    else:
        logger.error("Specify --synthetic or --real.")
        sys.exit(1)

    # Normalize modality tensors (z-score per feature)
    for mod in modality_tensors:
        t = modality_tensors[mod]
        mean = t.mean(dim=0, keepdim=True)
        std = t.std(dim=0, keepdim=True) + 1e-8
        t = (t - mean) / std
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        modality_tensors[mod] = t
    logger.info("Applied z-score normalization to %d modalities", len(modality_tensors))

    # Force graph rebuild when using real data (avoid stale cache)
    graph_cache = "data/graphs/ppi_pathway_graph.pt"
    if use_real:
        Path(graph_cache).unlink(missing_ok=True)

    builder = GraphBuilder(
        use_api=use_real,
        seed=args.seed,
        cache_path=graph_cache,
    )
    graph = builder.build(gene_ids, gene_names=gene_names)
    logger.info("Graph: %d nodes, %d edges", graph.num_nodes, graph.edge_index.shape[1])

    labels = build_labels(gene_names, gene_ids, seed=args.seed)
    n_pos = int(labels.sum().item())
    logger.info("Labels: %d positive, %d negative", n_pos, args.n_genes - n_pos)

    model = IrisModel(modality_features=MODALITY_FEATURES)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    trainer = IrisTrainer(model=model, lr=args.lr, epochs=args.epochs, seed=args.seed)
    history = trainer.train(modality_tensors, graph, labels)
    logger.info("Training complete. Final loss: %.4f", history["train_loss"][-1])

    model.eval()
    with torch.no_grad():
        scores = model(modality_tensors, graph)
    scores = torch.nan_to_num(scores, nan=0.0)

    evaluator = IrisEvaluator()
    metrics = evaluator.evaluate(scores, labels)
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    scores_path = Path(args.output_dir) / "iris_key_driver_scores.csv"
    eval_path = Path(args.output_dir) / "iris_evaluation.csv"

    ranked = evaluator.export_ranked_list(scores, gene_ids, labels)
    ensure_parent(scores_path)
    ranked.to_csv(scores_path, index=False)
    logger.info("Saved scores to %s", scores_path)

    metrics_df = pd.DataFrame([metrics])
    ensure_parent(eval_path)
    metrics_df.to_csv(eval_path, index=False)
    logger.info("Saved evaluation to %s", eval_path)

    logger.info("IRIS pipeline complete.")


if __name__ == "__main__":
    main()
