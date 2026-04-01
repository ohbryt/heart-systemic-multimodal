# IRIS Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve IRIS cardiac model by expanding epigenomics coverage, adding cross-validation, and integrating additional datasets.

**Architecture:** 
- Phase 1: Replace 34-gene snRNA DE with pseudobulk from h5ad (~17K genes)
- Phase 2: Add k-fold cross-validation to trainer
- Phase 3: Integrate additional datasets (GSE141910, HF sarcopenia)
- Phase 4: Run ablation study and enable STRING API

**Tech Stack:** Python 3.14, PyTorch, PyTorch Geometric, scanpy, pandas, numpy

---

## Phase 1: Improve Epigenomics Coverage (P0)

**Priority:** HIGH - Current epigenomics coverage is only 1.6% (34 genes)

### Task 1.1: Create Pseudobulk Extractor Function

**Files:**
- Modify: `src/data/cardiac_loader.py:185-226`
- Test: Add inline test in cardiac_loader.py

**Step 1: Read and understand h5ad structure**

```python
# Test loading h5ad to understand structure
import scanpy as sc
adata = sc.read_h5ad("/Users/ocm/.superset/worktrees/age related heart HF therapy/want-to-find-new-t/data/scrna_aging/processed/tms_heart_muscle_qc.h5ad")
print(adata)  # See shape, obs, var
print(adata.obs.columns)  # Check available metadata
print(adata.obs['cell_type'].unique() if 'cell_type' in adata.obs else adata.obs.keys())
```

**Step 2: Add pseudobulk function to cardiac_loader.py**

```python
def _load_pseudobulk_from_h5ad(
    h5ad_path: Path,
    groupby: str = "cell_type",
    age_col: str = "age",
) -> Tuple[List[str], torch.Tensor]:
    """Extract pseudobulk from h5ad: gene × (celltype × age_group) features.
    
    For each cell type × age group:
      - logFC: mean expression in older / younger
      - nlogp: -log10(p-value) from t-test
    
    Returns:
        gene_names: List of gene symbols
        features: Tensor of shape (n_genes, n_celltypes * 2)
    """
    import scanpy as sc
    
    adata = sc.read_h5ad(h5ad_path)
    
    # Get age groups (simplified: median split)
    median_age = adata.obs[age_col].median()
    adata.obs["age_group"] = adata.obs[age_col].apply(
        lambda x: "old" if x > median_age else "young"
    )
    
    # Compute pseudobulk per cell type × age group
    celltypes = adata.obs[groupby].unique()
    features_list = []
    feature_names = []
    
    for ct in celltypes:
        ct_data = adata[adata.obs[groupby] == ct]
        
        old_data = ct_data[ct_data.obs["age_group"] == "old"]
        young_data = ct_data[ct_data.obs["age_group"] == "young"]
        
        if old_data.n_obs < 5 or young_data.n_obs < 5:
            continue
        
        # Mean expression per group
        old_mean = old_data.X.mean(axis=0).A1 if hasattr(old_data.X, 'A') else old_data.X.mean(axis=0)
        young_mean = young_data.X.mean(axis=0).A1 if hasattr(young_data.X, 'A') else young_data.X.mean(axis=0)
        
        # logFC (pseudo)
        logfc = np.log2(old_mean + 1) - np.log2(young_mean + 1)
        
        # Simple p-value approximation using t-test statistic
        from scipy import stats
        _, pvals = stats.ttest_ind(old_data.X.A if hasattr(old_data.X, 'A') else old_data.X,
                                     young_data.X.A if hasattr(young_data.X, 'A') else young_data.X)
        nlogp = -np.log10(pvals + 1e-300)
        
        ct_short = ct.replace(" ", "_")[:10]
        features_list.append(np.column_stack([logfc, nlogp]))
        feature_names.extend([f"logFC_{ct_short}", f"nlogp_{ct_short}"])
    
    # Stack all features
    features = np.vstack(features_list).T  # (n_genes, n_features)
    gene_names = [g.upper() for g in adata.var_names.tolist()]
    
    return gene_names, torch.tensor(features, dtype=torch.float32)
```

**Step 3: Update _load_snrna_de to use pseudobulk**

Replace the current _load_snrna_de function to first try pseudobulk from h5ad, fallback to current CSV.

```python
def _load_snrna_de() -> Tuple[List[str], torch.Tensor]:
    """Load snRNA-seq pseudobulk features.
    
    Tries h5ad first, falls back to CSV if unavailable.
    """
    h5ad_path = CARDIAC_DATA_ROOT / "scrna_aging" / "processed" / "tms_heart_muscle_qc.h5ad"
    
    if h5ad_path.exists():
        logger.info("Loading pseudobulk from h5ad: %s", h5ad_path)
        try:
            return _load_pseudobulk_from_h5ad(h5ad_path)
        except Exception as e:
            logger.warning("Failed to load h5ad: %s, falling back to CSV", e)
    
    # Fallback to CSV
    path = CARDIAC_DATA_ROOT / "pressure_overload" / "results" / "snrnaseq_de_AS_vs_Control.csv"
    # ... existing code
```

**Step 4: Run test**

```bash
PYTHONPATH=. python -c "
from src.data.cardiac_loader import load_cardiac_data
data, genes, ids, feats = load_cardiac_data(max_genes=500, seed=42)
for name, d in data.items():
    coverage = 100 * (d.features.abs().sum(dim=1) > 0).float().mean().item()
    print(f'{name}: {d.features.shape[0]} genes × {d.features.shape[1]} features, coverage {coverage:.1f}%')
"
```

Expected: epigenomics coverage should be >40% (vs current 1.6%)

---

## Phase 2: Add K-Fold Cross-Validation (P1)

**Priority:** HIGH - Publication credibility requires CV

### Task 2.1: Add Cross-Validation to Trainer

**Files:**
- Modify: `src/training/trainer.py`
- Test: `tests/test_trainer.py` (create if not exists)

**Step 1: Write failing test for cross-validation**

```python
# tests/test_trainer.py
import pytest
import torch
from src.models.iris_model import IrisModel
from src.training.trainer import IrisTrainer
from src.training.evaluator import IrisEvaluator
from torch_geometric.data import Data

def test_cross_validation_returns_fold_metrics():
    """CV should return metrics for each fold."""
    # Create dummy data
    n_genes = 200
    modality_features = {"bulk_rna": 64, "proteomics": 50}
    
    model = IrisModel(modality_features=modality_features)
    
    # Dummy graph
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    graph = Data(edge_index=edge_index, num_nodes=n_genes)
    
    # Dummy data
    modality_data = {
        "bulk_rna": torch.randn(n_genes, 64),
        "proteomics": torch.randn(n_genes, 50),
    }
    labels = torch.zeros(n_genes)
    labels[:20] = 1  # 20 positives
    
    trainer = IrisTrainer(model=model, epochs=5, patience=5)
    
    # This method should exist
    results = trainer.cross_validate(modality_data, graph, labels, k=3)
    
    assert "fold_metrics" in results
    assert len(results["fold_metrics"]) == 3
    assert "auroc" in results["fold_metrics"][0]
    assert "auprc" in results["fold_metrics"][0]
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. python -m pytest tests/test_trainer.py::test_cross_validation_returns_fold_metrics -v
```

Expected: FAIL with "AttributeError: 'IrisTrainer' object has no attribute 'cross_validate'"

**Step 3: Implement cross-validation in trainer.py**

Add to `src/training/trainer.py`:

```python
def cross_validate(
    self,
    modality_data: Dict[str, torch.Tensor],
    graph: Data,
    labels: torch.Tensor,
    k: int = 5,
) -> Dict[str, any]:
    """K-fold cross-validation.
    
    Returns:
        {
            "fold_metrics": [{auroc, auprc, ndcg_20, ...}, ...],
            "mean_metrics": {auroc_mean, auroc_std, ...},
            "all_predictions": [(scores, labels, fold_idx), ...],
        }
    """
    from sklearn.model_selection import KFold
    from src.training.evaluator import IrisEvaluator
    
    torch.manual_seed(self.seed)
    n = labels.shape[0]
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=self.seed)
    evaluator = IrisEvaluator()
    
    fold_metrics = []
    all_predictions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(torch.arange(n))):
        logger.info(f"=== Fold {fold_idx + 1}/{k} ===")
        
        # Clone model for each fold
        model = self.model.__class__(**self._get_model_config())
        
        # Compute pos_weight
        n_pos = labels[train_idx].sum().item()
        n_neg = len(train_idx) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        
        optimizer = AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            scores = model(modality_data, graph)
            
            loss = model.ranking_head.compute_loss(
                scores[train_idx], labels[train_idx], pos_weight=pos_weight
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_scores = model(modality_data, graph)[val_idx]
                val_loss = model.ranking_head.compute_loss(
                    val_scores, labels[val_idx], pos_weight=pos_weight
                )
            
            scheduler.step(val_loss.item())
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Evaluate on validation fold
        model.eval()
        with torch.no_grad():
            val_scores = model(modality_data, graph)[val_idx]
            fold_result = evaluator.evaluate(val_scores, labels[val_idx])
            fold_metrics.append(fold_result)
            all_predictions.append((val_scores, labels[val_idx], fold_idx))
            
            logger.info(
                f"Fold {fold_idx + 1}: AUROC={fold_result['auroc']:.3f}, AUPRC={fold_result['auprc']:.3f}"
            )
    
    # Compute mean ± std
    mean_metrics = {}
    for key in fold_metrics[0].keys:
        values = [fm[key] for fm in fold_metrics]
        mean_metrics[f"{key}_mean"] = np.mean(values)
        mean_metrics[f"{key}_std"] = np.std(values)
    
    logger.info(f"CV Results: AUROC={mean_metrics['auroc_mean']:.3f}±{mean_metrics['auroc_std']:.3f}")
    
    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "all_predictions": all_predictions,
    }

def _get_model_config(self) -> Dict:
    """Get model config for recreation."""
    # Extract from existing model
    mod_features = {}
    for name in self.model.modality_names:
        mod_features[name] = self.model.encoder_bank.encoders[name].input_dim
    
    return {
        "modality_features": mod_features,
        "embed_dim": self.model.encoder_bank.out_dim,
        "fused_dim": self.model.fusion.fused_dim,
        "gnn_hidden": self.model.gnn.hidden_dim,
        "gnn_out": self.model.gnn.out_dim,
        "gnn_heads": self.model.gnn.heads,
        "gnn_layers": self.model.gnn.layers,
        "ranking_hidden": self.model.ranking_head.mlp[0].out_features,
    }
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest tests/test_trainer.py::test_cross_validation_returns_fold_metrics -v
```

**Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat: add k-fold cross-validation to trainer"
```

### Task 2.2: Update run script to use CV

**Files:**
- Modify: `scripts/run_iris_model.py`

**Step 1: Add --cv flag**

```python
# In run_iris_model.py, add argument
parser.add_argument("--cv", type=int, default=0, help="K-fold CV (0 for single split)")
```

**Step 2: Use CV when specified**

```python
if args.cv > 1:
    logger.info(f"Running {args.cv}-fold cross-validation...")
    results = trainer.cross_validate(
        modality_tensors, graph, labels, k=args.cv
    )
    # Print mean ± std
    for key, mean in results["mean_metrics"].items():
        if "_std" not in key:
            std = results["mean_metrics"].get(f"{key}_std", 0)
            logger.info(f"  {key}: {mean:.3f} ± {std:.3f}")
else:
    # Existing single split training
    history = trainer.train(modality_tensors, graph, labels)
```

---

## Phase 3: Integrate Additional Datasets (P2, P3)

### Task 3.1: Integrate GSE141910

**Files:**
- Modify: `src/data/cardiac_loader.py`

**Step 1: Add GSE141910 loader**

```python
def _load_gse141910() -> Tuple[List[str], torch.Tensor]:
    """Load GSE141910 bulk RNA (50+ cardiac samples).
    
    Each sample is a separate gzipped CSV.
    """
    base_path = CARDIAC_DATA_ROOT / "geo" / "GSE141910"
    
    gene_expr = {}
    sample_names = []
    
    for csv_file in sorted(base_path.glob("*.csv.gz")):
        sample_name = csv_file.stem.replace(".csv", "")
        sample_names.append(sample_name)
        
        df = pd.read_csv(csv_file)
        # Assume columns: gene, expression
        for _, row in df.iterrows():
            gene = row.get("gene", row.get("Gene", "")).upper()
            expr = row.get("expression", row.get("expr", row.get("value", 0)))
            if gene:
                if gene not in gene_expr:
                    gene_expr[gene] = []
                gene_expr[gene].append(float(expr))
    
    # Align to common genes
    gene_names = sorted(gene_expr.keys())
    mat = np.array([gene_expr[g] for g in gene_names], dtype=np.float32)
    
    logger.info("  GSE141910: %d genes × %d samples", len(gene_names), len(sample_names))
    return gene_names, torch.tensor(mat, dtype=torch.float32)
```

**Step 2: Add to load_cardiac_data**

```python
# After loading existing modalities, add:
gse141910_genes, gse141910_mat = _load_gse141910()
# Add as additional modality or concat with bulk_rna
```

---

## Phase 4: Ablation Study (P4)

### Task 4.1: Create Ablation Script

**Files:**
- Create: `scripts/run_ablation_study.py`

**Step 1: Write ablation script**

```python
"""Ablation study: quantify each modality's contribution."""
import argparse
import torch
from src.data.cardiac_loader import load_cardiac_data
from src.data.graph_builder import build_graph
from src.data.cardiac_labels import load_cardiac_labels
from src.models.iris_model import IrisModel
from src.training.trainer import IrisTrainer
from src.training.evaluator import IrisEvaluator

def run_ablation(modality_data, graph, labels, exclude_modality=None):
    """Train with specified modality excluded."""
    if exclude_modality:
        data = {k: v for k, v in modality_data.items() if k != exclude_modality}
    else:
        data = modality_data
    
    model = IrisModel(modality_features={k: v.shape[1] for k, v in data.items()})
    trainer = IrisTrainer(model=model, epochs=100, patience=20)
    history = trainer.train(data, graph, labels)
    
    model.eval()
    with torch.no_grad():
        scores = model(data, graph)
    
    evaluator = IrisEvaluator()
    return evaluator.evaluate(scores, labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-genes", type=int, default=2000)
    args = parser.parse_args()
    
    # Load data
    data, genes, _, _ = load_cardiac_data(max_genes=args.n_genes)
    graph = build_graph(genes, use_api=False)
    labels = load_cardiac_labels(genes)
    
    modality_tensors = {k: v.features for k, v in data.items()}
    
    # Baseline (all modalities)
    print("=== Baseline (all modalities) ===")
    baseline = run_ablation(modality_tensors, graph, labels)
    print(f"AUROC: {baseline['auroc']:.3f}, AUPRC: {baseline['auprc']:.3f}")
    
    # Leave-one-out
    for mod in modality_tensors.keys():
        print(f"\n=== Without {mod} ===")
        result = run_ablation(modality_tensors, graph, labels, exclude_modality=mod)
        print(f"AUROC: {result['auroc']:.3f}, AUPRC: {result['auprc']:.3f}")
        print(f"  ΔAUROC: {result['auroc'] - baseline['auroc']:.3f}")

if __name__ == "__main__":
    main()
```

**Step 2: Run ablation**

```bash
PYTHONPATH=. python scripts/run_ablation_study.py --n-genes 1000
```

---

## Phase 5: Enable STRING API (P5)

### Task 5.1: Update Cardiac Mode to Use Real PPI

**Files:**
- Modify: `scripts/run_iris_model.py` or `src/data/graph_builder.py`

**Step 1: Change use_api flag**

In run_iris_model.py, find the cardiac mode path and change:

```python
# Currently: graph = build_graph(genes, use_api=False)
# Change to:
graph = build_graph(genes, use_api=True)  # Requires internet, ~30 seconds
```

---

## Summary

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| P0: Improve epigenomics | HIGH | 2h | High (1.6% → 40%+) |
| P1: Cross-validation | HIGH | 3h | High (publication credibility) |
| P2: GSE141910 | MEDIUM | 2h | Medium |
| P3: HF Sarcopenia | MEDIUM | 2h | Medium |
| P4: Ablation Study | LOW | 1h | Medium |
| P5: STRING API | LOW | 30min | Low |

**Total estimated time:** ~10-12 hours

---

## Execution Options

**Plan complete and saved to `docs/plans/2026-04-01-iris-improvement-plan.md`. Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
