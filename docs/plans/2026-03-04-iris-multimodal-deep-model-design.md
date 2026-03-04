# IRIS: Multimodal Deep Model for Sarcopenia Key Driver Discovery

## Objective

Build a multimodal deep learning model that integrates all 6 omics modalities (bulk RNA, scRNA, spatial transcriptomics, proteomics, metabolomics, epigenomics) through modality-specific encoders fused on a PPI+pathway graph to rank sarcopenia key driver genes.

## Approach

**Modality-Specific Encoders → GNN Fusion → Key Driver Ranking**

Each modality gets its own MLP encoder projecting features into a shared embedding space. Per-gene embeddings are placed on PPI+pathway graph nodes and refined through GAT layers. Final node embeddings feed a ranking head for key driver scoring.

## Architecture

```
  bulk_rna    scrna    spatial   proteomics   metabolomics   epigenomics
  Encoder    Encoder   Encoder    Encoder       Encoder        Encoder
     │          │         │          │              │              │
     └──────────┴─────────┴──────────┴──────────────┴──────────────┘
                                  │
                    Modality Fusion (concat + project)
                    + missing-modality mask
                                  │
                    PPI + Pathway Graph (STRING/KEGG)
                    GAT layers (2-3, 4 heads)
                                  │
                    Key Driver Ranking Head
                    (MLP → score per gene node)
```

## Data Layer

### Synthetic Data Generator (`src/data/synthetic_generator.py`)
- Reads Wave 0 dataset registry for real gene lists and modality metadata
- Generates realistic synthetic feature matrices per modality (log-normal for RNA, beta for methylation, sparse for proteomics)
- Outputs standardized `ModalityData` objects: `(n_genes, n_features)` with shared gene ID index

### Real Data Loader Interface (`src/data/loaders.py`)
- Abstract `ModalityLoader` base class: `load(dataset_id) → ModalityData`
- Concrete loaders slot in when A3 delivers real matrices
- Same `ModalityData` contract — model code is data-source agnostic

### Graph Builder (`src/data/graph_builder.py`)
- STRING API PPI edges (confidence ≥ 700)
- KEGG REST API pathway membership
- Pathway co-membership edges (genes sharing ≥ 2 pathways)
- Outputs PyG `Data` object with edge types: `ppi` and `pathway`
- Cached to `data/graphs/ppi_pathway_graph.pt`

### Gene ID Alignment (`src/data/gene_mapper.py`)
- Canonical ID space: Entrez gene IDs
- Mapping: symbol → Entrez, Ensembl → Entrez, UniProt → Entrez
- Unmappable genes dropped with warning

## Model Components

### Modality Encoders (`src/models/encoders.py`)
- `ModalityEncoder(in_features, hidden_dim=64, out_dim=128)`
- 2-layer MLP + BatchNorm + ReLU + Dropout(0.3)
- One per modality in a `ModuleDict`

### Fusion Module (`src/models/fusion.py`)
- `ModalityFusion(n_modalities=6, embed_dim=128, fused_dim=128)`
- Concatenates available embeddings, zero-masks missing modalities
- Learned modality-presence vector as additive bias
- Linear projection to fused dimension

### GNN Backbone (`src/models/gnn.py`)
- `PPIPathwayGNN(in_dim=128, hidden_dim=128, out_dim=128, heads=4, layers=3)`
- 3-layer GATv2Conv, multi-head attention, residual connections, LayerNorm
- Edge-type features for PPI vs pathway distinction

### Ranking Head (`src/models/ranking_head.py`)
- `KeyDriverRankingHead(in_dim=128, hidden_dim=64)`
- 2-layer MLP → scalar score per gene
- MarginRankingLoss on (positive, negative) gene pairs

### Full Model (`src/models/iris_model.py`)
- `IrisModel`: encoders → fusion → GNN → ranking head
- `forward(modality_data: Dict[str, Tensor], graph: Data) → Tensor`

## Training

- **Supervision**: Literature evidence scores mapped to genes (high evidence = positive, low/absent = negative)
- **Loss**: MarginRankingLoss
- **Optimizer**: AdamW, lr=1e-3, cosine annealing
- **Split**: 80/20 stratified by evidence tier
- **Early stopping**: patience=10 on validation ranking loss

## Evaluation

- **Metrics**: AUROC, AUPRC, NDCG@k (k=20, 50, 100), Hit@k
- **Baseline**: existing `key_driver_model.py` as non-GNN comparison
- **Output**: `data/reports/iris_evaluation.csv`

## Integration

- Entry script: `scripts/run_iris_model.py`
- Output artifact: `data/reports/iris_key_driver_scores.csv` (gene, score, rank, evidence_tier)
- Replaces A5-DeepLearning placeholder in `configs/subagent_architecture.yaml`
- Candidate board reads iris scores alongside/instead of existing key driver scores

## Testing

- `scripts/test_iris_model.sh` — end-to-end on synthetic data
- Unit tests per component: encoders, fusion, GNN, ranking head, graph builder

## Tech Stack

- PyTorch + PyTorch Geometric (PyG)
- GATv2Conv for graph attention
- STRING API + KEGG REST API for graph construction
- pandas for data handling
- Existing project utilities (`src/utils/`)

## File Structure

```
src/
  data/
    synthetic_generator.py
    loaders.py
    graph_builder.py
    gene_mapper.py
  models/
    encoders.py
    fusion.py
    gnn.py
    ranking_head.py
    iris_model.py
  training/
    trainer.py
    evaluator.py
scripts/
  run_iris_model.py
  test_iris_model.sh
data/
  graphs/
    ppi_pathway_graph.pt
  reports/
    iris_key_driver_scores.csv
    iris_evaluation.csv
```
