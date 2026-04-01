# IRIS Cardiac — Continuation Guide

> For continuing this project in OpenCode or any other AI coding assistant.
> Last updated: 2026-04-01

## Project Overview

**IRIS** (Integrative Ranking of Interventional Scores) is a multimodal deep learning pipeline that ranks genes by their importance as key drivers of disease using multi-omics data + GNN-based graph reasoning.

**Architecture:** Per-modality encoders → Attention fusion → GNN (GATv2 over PPI+pathway graph) → Ranking head (BCE loss)

**Branch:** `iris` (worktree at `/Users/ocm/.superset/worktrees/muscle_diff/iris`)

---

## Current State (2026-04-01)

### What Works

```bash
# Activate environment
source .venv/bin/activate

# Synthetic mode (500 genes, ~2 seconds)
PYTHONPATH=. python scripts/run_iris_model.py --synthetic

# Cardiac mode (2000 genes, real data, ~10 seconds)
PYTHONPATH=. python scripts/run_iris_model.py --cardiac --n-genes 2000

# Tests (33/33 passing)
PYTHONPATH=. python -m pytest tests/ -q
```

### Latest Cardiac Metrics

| Metric | Value |
|--------|-------|
| AUROC | 0.922 |
| AUPRC | 0.732 |
| nDCG@20 | 1.000 |
| nDCG@50 | 0.859 |
| nDCG@100 | 0.806 |
| Hit@50 | 0.556 |

### Top Novel Candidates (not in label set, ranked by IRIS score)

| Rank | Gene | Score | Known Role |
|------|------|-------|-----------|
| 38 | OSM | 2.46 | IL-6 family, cardiac fibrosis |
| 41 | PDGFB | 2.37 | Vascular remodeling |
| 42 | NRP1 | 2.31 | VEGF co-receptor, angiogenesis |
| 43 | CRYAB | 2.28 | Cardiomyocyte stress response |
| 44 | MB | 2.24 | Cardiac O2 transport |
| 45 | SERPINE1 | 2.19 | PAI-1, thrombosis/fibrosis |
| 47 | HGF | 2.07 | Cardioprotective |

---

## Data Sources (Cardiac Mode)

All data lives on this Mac Mini. Paths are hardcoded in `src/data/cardiac_loader.py`.

| Modality | Source | Path (relative to CARDIAC_DATA_ROOT) | Genes | Features | Coverage |
|----------|--------|--------------------------------------|-------|----------|----------|
| bulk_rna | GSE116250 LV RPKM | `geo/GSE116250/GSE116250_rpkm.txt.gz` | 55,826 | 64 samples (NF/DCM/ICM) | 97.9% |
| proteomics | Olink Explore 3072 | `pressure_overload/processed/olink_proteomics_gene_symbols.csv` | 979 | 809 patients | 48.9% |
| scrna | TMS aging DE | `scrna_aging/processed/aging_de_per_celltype.csv` | 17,116 | 18 (9 celltypes × logFC+nlogp) | 90.9% |
| epigenomics | snRNA AS vs Ctrl DE | `pressure_overload/results/snrnaseq_de_AS_vs_Control.csv` | 34 | 28 (7 celltypes × 4 stats) | **1.6%** |

**CARDIAC_DATA_ROOT:** `/Users/ocm/.superset/worktrees/age related heart HF therapy/want-to-find-new-t/data`

### Additional Available Datasets (Not Yet Integrated)

| Dataset | Location | Description |
|---------|----------|-------------|
| GSE141910 | `geo/GSE141910/` | 50+ cardiac samples, gzipped CSVs per sample |
| TMS heart h5ad | `scrna_aging/processed/tms_heart_muscle_qc.h5ad` | 149 MB, full single-cell matrix (needs scanpy) |
| GSE278576 | `buttoned-laugh/data/raw/expression/` | 10x H5 files, 4 healthy donors young vs old |
| HF sarcopenia DEGs | `/Users/ocm/.superset/worktrees/drug-discovery-pipeline/euphonium/HF_sarcopenia_project/results/tables/` | 1000 DEGs from GSE57345 (313 HF patients) |
| Olink raw (OID format) | `pressure_overload/processed/olink_proteomics_processed.csv` | 809 patients × 981 OID columns |
| OID→Gene mapping | `pressure_overload/processed/oid_gene_mapping.csv` | Maps OID codes to gene symbols |

---

## Codebase Map

```
iris/
├── scripts/
│   └── run_iris_model.py          # Entry point: --synthetic | --real | --cardiac
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py  # Synthetic data (ModalityData dataclass defined here)
│   │   ├── cardiac_loader.py       # Loads real cardiac datasets → ModalityData
│   │   ├── cardiac_labels.py       # ~160 curated HF/cardiac remodeling genes
│   │   ├── label_builder.py        # ~80 curated sarcopenia genes (original)
│   │   └── graph_builder.py        # PPI (STRING API) + pathway edges (KEGG + cardiac)
│   ├── models/
│   │   ├── iris_model.py           # Full pipeline: encoders → fusion → GNN → ranking
│   │   ├── encoders.py             # Per-modality Linear→ReLU→Linear→LayerNorm
│   │   ├── fusion.py               # Attention-based modality fusion
│   │   ├── gnn.py                  # GATv2Conv, 3 layers, 4 heads
│   │   └── ranking_head.py         # MLP → scalar score, BCEWithLogitsLoss
│   ├── training/
│   │   ├── trainer.py              # Train loop, ReduceLROnPlateau, early stopping
│   │   └── evaluator.py            # AUROC, AUPRC, nDCG@k, Hit@k
│   └── utils/
│       ├── io_utils.py
│       ├── http_utils.py           # Retry logic for STRING API
│       └── schema.py
├── tests/                          # 33 tests, all passing
├── data/
│   ├── graphs/ppi_pathway_graph.pt # Cached graph (auto-rebuilt for cardiac)
│   └── reports/
│       ├── iris_key_driver_scores.csv  # gene_name, gene_id, score, rank, evidence_tier
│       └── iris_evaluation.csv         # auroc, auprc, ndcg_20, ndcg_50, ndcg_100, hit_20, hit_50
├── configs/
├── DATASETS.md                     # Full dataset catalog
└── continue.md                     # This file
```

### Key Classes and Interfaces

**ModalityData** (`src/data/synthetic_generator.py`):
```python
@dataclass
class ModalityData:
    features: torch.Tensor  # (n_genes, n_features)
    gene_ids: List[int]
    modality: str
    gene_names: List[str] | None = None
```

**IrisModel** (`src/models/iris_model.py`):
```python
model = IrisModel(modality_features={"bulk_rna": 64, "proteomics": 809, ...})
scores = model(modality_tensors, graph)  # → (n_genes,) tensor
```

**GraphBuilder** (`src/data/graph_builder.py`):
- Uses `MUSCLE_PATHWAY_GENES` + `CARDIAC_PATHWAY_GENES` dicts
- Builds synthetic PPI when `use_api=False`
- Pathway edges connect all co-pathway genes (clique)

---

## Priority Tasks

### P0: Improve Epigenomics Coverage (Currently 1.6%)

The snRNA DE results only have 34 genes. Fix by extracting pseudobulk from the full h5ad:

```bash
pip install scanpy
```

```python
import scanpy as sc
adata = sc.read_h5ad("path/to/tms_heart_muscle_qc.h5ad")
# Pseudobulk per cell type × age group → gene × feature matrix
# This gives ~17K gene coverage instead of 34
```

Update `cardiac_loader.py::_load_snrna_de()` to use pseudobulk when h5ad is available.

### P1: Cross-Validation

Currently single random split. Add k-fold CV:
- In `trainer.py`, add a `cross_validate(k=5)` method
- Report mean ± std for all metrics
- Important for publication credibility

### P2: Integrate GSE141910 (50+ cardiac samples)

Additional bulk RNA data. Each sample is a separate gzipped CSV in `geo/GSE141910/`.
- Parse all ~50 CSVs, merge into expression matrix
- Concat with GSE116250 features (65 + 50 = 115 bulk RNA features)
- Or treat as a separate "bulk_rna_2" modality

### P3: Integrate HF Sarcopenia Multi-Omics

From `euphonium/HF_sarcopenia_project/results/tables/`:
- `transcriptomic_DEGs.csv` — 1000 genes with logFC, p-values
- Add as additional features or a new modality

### P4: Ablation Study

Quantify each modality's contribution:
- Train with all 4 modalities (baseline)
- Train with each modality removed (leave-one-out)
- Train with single modalities
- Compare AUROC/AUPRC

### P5: STRING API Integration for Cardiac Mode

Currently `use_api=False` for cardiac mode (synthetic PPI). Enable real STRING PPI:
- Set `use_api=True` in the cardiac path
- This will query STRING-DB for the 2000 genes
- Takes ~30 seconds, needs internet

---

## Known Issues

1. **Graph cache stale risk**: Cache at `data/graphs/ppi_pathway_graph.pt` is auto-cleared for cardiac/real modes, but if you change `--n-genes`, delete the cache manually.

2. **pos_weight imbalance**: 72/2000 positive = 3.6%. The BCEWithLogitsLoss uses `pos_weight=28.63`. This is aggressive — if performance is unstable, try capping at 10.

3. **Gene ID mapping**: `gene_id` in output CSV is a sequential integer (1-based), not Entrez. Use `gene_name` column for biological interpretation.

4. **Epigenomics mislabel**: The `epigenomics` modality slot currently holds snRNA DE features (disease context), not true epigenomic data. Rename if this causes confusion.

5. **Python 3.14 warnings**: PyTorch JIT deprecation warnings on Python 3.14. Harmless but noisy.

---

## Environment

```bash
# Python 3.14, venv at .venv/
source .venv/bin/activate
# Key deps: torch, torch_geometric, pandas, numpy
# All in requirements.txt

# Always set PYTHONPATH for imports
export PYTHONPATH=.
```

## Git

```bash
# Branch: iris
# Latest commit: 7def8e9 (feat: improve IRIS training with literature-based labels and BCE loss)
# Uncommitted: cardiac_loader.py, cardiac_labels.py, graph_builder.py updates, run script updates
```
