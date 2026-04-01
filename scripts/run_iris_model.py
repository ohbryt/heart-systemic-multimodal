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
from src.data.cardiac_labels import build_cardiac_labels
from src.data.cardiac_loader import load_cardiac_data
from src.models.iris_model import IrisModel
from src.training.evaluator import IrisEvaluator
from src.training.trainer import IrisTrainer
from src.utils.io_utils import ensure_parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="IRIS multimodal deep model")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument(
        "--real", action="store_true", help="Use real data via A3 ingestion"
    )
    parser.add_argument("--cardiac", action="store_true", help="Use cardiac data")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use STRING API for PPI graph (requires internet)",
    )
    parser.add_argument("--n-genes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/reports")
    args = parser.parse_args()

    use_real = args.real and not args.synthetic
    use_cardiac = args.cardiac
    use_api = args.use_api or use_real
    mode = "cardiac" if use_cardiac else ("real" if use_real else "synthetic")
    logger.info("IRIS pipeline starting (mode=%s, n_genes=%d)", mode, args.n_genes)

    gene_names: list[str] | None = None
    cardiac_modality_features: dict[str, int] | None = None

    if use_cardiac:
        data, gene_names, gene_ids, cardiac_modality_features = load_cardiac_data(
            max_genes=args.n_genes,
            seed=args.seed,
        )
        modality_tensors = {mod: data[mod].features for mod in data}
        args.n_genes = len(gene_ids)
    elif use_real:
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
        modality_tensors = {
            mod: data[mod].features for mod in MODALITY_FEATURES if mod in data
        }
        args.n_genes = len(gene_ids)
    elif args.synthetic:
        gen = SyntheticDataGenerator(n_genes=args.n_genes, seed=args.seed)
        data = gen.generate()
        gene_ids = data["bulk_rna"].gene_ids
        gene_names = data["bulk_rna"].gene_names
        modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    else:
        logger.error("Specify --synthetic, --real, or --cardiac.")
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

    # Force graph rebuild when using real/cardiac data (avoid stale cache)
    graph_cache = "data/graphs/ppi_pathway_graph.pt"
    if use_real or use_cardiac:
        Path(graph_cache).unlink(missing_ok=True)

    builder = GraphBuilder(
        use_api=use_api,
        seed=args.seed,
        cache_path=graph_cache,
    )
    graph = builder.build(gene_ids, gene_names=gene_names)
    logger.info("Graph: %d nodes, %d edges", graph.num_nodes, graph.edge_index.shape[1])

    if use_cardiac:
        labels = build_cardiac_labels(gene_names, gene_ids, seed=args.seed)
    else:
        labels = build_labels(gene_names, gene_ids, seed=args.seed)
    n_pos = int(labels.sum().item())
    logger.info("Labels: %d positive, %d negative", n_pos, args.n_genes - n_pos)

    active_modality_features = (
        cardiac_modality_features if use_cardiac else MODALITY_FEATURES
    )
    model = IrisModel(modality_features=active_modality_features)
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
    if gene_names:
        ranked["gene_name"] = [gene_names[i] for i in ranked["gene_id"] - 1]
        cols = ["gene_name"] + [c for c in ranked.columns if c != "gene_name"]
        ranked = ranked[cols]
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
