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
from src.data.synthetic_generator import MODALITY_FEATURES, SyntheticDataGenerator
from src.models.iris_model import IrisModel
from src.training.evaluator import IrisEvaluator
from src.training.trainer import IrisTrainer
from src.utils.io_utils import ensure_parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _make_labels(n_genes: int, seed: int = 42) -> torch.Tensor:
    """Create supervision labels. In production, map from literature registry."""
    rng = torch.Generator().manual_seed(seed)
    labels = torch.zeros(n_genes)
    n_pos = max(1, n_genes // 5)
    pos_idx = torch.randperm(n_genes, generator=rng)[:n_pos]
    labels[pos_idx] = 1.0
    return labels


def main():
    parser = argparse.ArgumentParser(description="IRIS multimodal deep model")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--n-genes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/reports")
    args = parser.parse_args()

    logger.info("IRIS pipeline starting (synthetic=%s, n_genes=%d)", args.synthetic, args.n_genes)

    if args.synthetic:
        gen = SyntheticDataGenerator(n_genes=args.n_genes, seed=args.seed)
        data = gen.generate()
        gene_ids = data["bulk_rna"].gene_ids
        modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    else:
        logger.error("Real data mode requires A3 ingestion pipeline. Use --synthetic.")
        sys.exit(1)

    graph_cache = "data/graphs/ppi_pathway_graph.pt"
    builder = GraphBuilder(
        use_api=not args.synthetic,
        seed=args.seed,
        cache_path=graph_cache,
    )
    graph = builder.build(gene_ids)
    logger.info("Graph: %d nodes, %d edges", graph.num_nodes, graph.edge_index.shape[1])

    labels = _make_labels(args.n_genes, args.seed)
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
