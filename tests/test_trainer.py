import pytest
import torch

from src.training.trainer import IrisTrainer
from src.models.iris_model import IrisModel
from src.data.synthetic_generator import SyntheticDataGenerator, MODALITY_FEATURES
from src.data.graph_builder import GraphBuilder


def test_trainer_runs_one_epoch():
    n_genes = 100
    gen = SyntheticDataGenerator(n_genes=n_genes, seed=42)
    data = gen.generate()
    graph = GraphBuilder(use_api=False, seed=42).build(data["bulk_rna"].gene_ids)
    model = IrisModel(modality_features=MODALITY_FEATURES)

    labels = torch.zeros(n_genes)
    labels[:30] = 1.0

    trainer = IrisTrainer(model=model, lr=1e-3, epochs=1)
    modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    history = trainer.train(modality_tensors, graph, labels)

    assert "train_loss" in history
    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0


def test_trainer_loss_decreases():
    n_genes = 100
    gen = SyntheticDataGenerator(n_genes=n_genes, seed=42)
    data = gen.generate()
    graph = GraphBuilder(use_api=False, seed=42).build(data["bulk_rna"].gene_ids)
    model = IrisModel(modality_features=MODALITY_FEATURES, gnn_layers=1)

    labels = torch.zeros(n_genes)
    labels[:30] = 1.0

    trainer = IrisTrainer(model=model, lr=1e-2, epochs=20)
    modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    history = trainer.train(modality_tensors, graph, labels)

    first_5_avg = sum(history["train_loss"][:5]) / 5
    last_5_avg = sum(history["train_loss"][-5:]) / 5
    assert last_5_avg < first_5_avg, "Loss did not decrease over training"
