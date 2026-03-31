# Heart-Systemic Multi-modal Analysis Pipeline

A modular, RAM-aware Python pipeline for integrating single-cell RNA-seq,
bulk RNA-seq, and proteomics data to identify **cardiac secretome candidates**
and **systemic cross-tissue responders** in heart failure.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Datasets](#datasets)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Manual Data Placement](#manual-data-placement)
7. [Pipeline Steps](#pipeline-steps)
8. [Output Structure](#output-structure)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

Heart failure is a systemic disease — the failing heart sends signals
(cardiokines, secreted proteins) that alter gene expression in remote tissues
such as liver and skeletal muscle.  This pipeline:

1. Integrates multi-modal cardiac datasets (scRNA-seq + proteomics + bulk RNA-seq).
2. Scores cells for fibrosis, ECM remodeling, inflammation, hypertrophy,
   mitochondrial stress, senescence, and secretory programs.
3. Identifies candidate secreted proteins (secretome) by combining expression
   evidence with signal-peptide prediction databases.
4. Performs cross-tissue overlap analysis to find genes co-regulated in
   heart and systemic tissues.
5. Identifies systemic responder cell populations correlated with cardiac
   disease severity.
6. Ranks candidates by a composite score weighting multiple evidence layers.

---

## Datasets

### Single-cell / Single-nucleus RNA-seq

| Key | Accession | Description |
|-----|-----------|-------------|
| `heart_cellxgene` | CELLxGENE Census | Human heart cells — normal + DCM/ICM/HF |
| `heart_failure_GSE183852` | GSE183852 | Human HF snRNA-seq (DCM/ICM vs healthy) |
| `cardiac_fibrosis_GSE135805` | GSE135805 | Human cardiac fibrosis scRNA-seq |
| `heart_aging_GSE290577` | GSE290577 | Human aging heart scRNA-seq |

### Proteomics (PRIDE Archive)

| Key | Accession | Description |
|-----|-----------|-------------|
| `heart_proteomics_PXD021371` | PXD021371 | Heart failure plasma/tissue proteomics |
| `secretome_PXD059929` | PXD059929 | Cardiac secretome (conditioned media) |
| `cross_tissue_PXD060680` | PXD060680 | Cross-tissue proteomics (heart/liver/muscle) |

### Bulk RNA-seq

| Key | Accession | Description |
|-----|-----------|-------------|
| `emtab_15659` | E-MTAB-15659 | Cross-tissue bulk RNA-seq (ArrayExpress) |

### Responder Datasets (placeholders)

| Key | Description |
|-----|-------------|
| `liver_responder` | Liver snRNA-seq from HF patients — enable when data available |
| `muscle_responder` | Skeletal muscle snRNA-seq from HF patients |
| `custom_responder` | User-defined tissue dataset |

---

## Setup

### Prerequisites

- Python 3.11 or newer
- ~16 GB RAM recommended (32 GB for full dataset runs)
- ~50 GB free disk space for raw + processed data

### Installation

```bash
# Clone the repository
git clone https://github.com/brownbiotech/heart-systemic-multimodal.git
cd heart-systemic-multimodal

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install the package (core dependencies)
pip install -e .

# Optional: install CELLxGENE Census support
pip install -e ".[census]"

# Optional: install everything including dev tools
pip install -e ".[full]"
```

### Verify installation

```bash
python main.py --version
# or, after pip install:
hsm --version
```

---

## Configuration

### Default config

All settings live in `config/default_config.yaml`.  The defaults are
designed to work out-of-the-box once data is placed in the expected
directories (see [Manual Data Placement](#manual-data-placement)).

### User overrides

Create a YAML file with only the keys you want to change:

```yaml
# my_config.yaml
paths:
  data_dir: /mnt/data/heart_project/raw
  results_dir: /mnt/data/heart_project/results

preprocessing:
  downsample_to: 100000    # override to 100k cells
  n_top_genes: 4000

execution:
  n_workers: 8
  log_level: DEBUG
```

Pass it to any command with `--config`:

```bash
hsm preprocess --config my_config.yaml
```

### Key configuration parameters

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `preprocessing` | `n_top_genes` | 3000 | Highly variable genes |
| `preprocessing` | `min_cells` | 3 | Min cells per gene |
| `preprocessing` | `downsample_to` | 50000 | Max cells per dataset (null = off) |
| `preprocessing` | `max_mito_fraction` | 0.20 | QC: max MT gene fraction |
| `analysis.ranking` | `top_n` | 20 | Top candidates per category |
| `execution` | `n_workers` | 4 | Parallel workers |
| `execution` | `use_cache` | true | Cache intermediate results |

---

## Usage

### Full pipeline (recommended first run)

```bash
# Download all enabled datasets, then run full pipeline
hsm run-all --config my_config.yaml

# Dry-run to preview without executing
hsm run-all --dry-run
```

### Individual commands

```bash
# 1. Download raw data
hsm download
hsm download --dataset GSE183852 --dataset GSE135805   # specific datasets

# 2. Preprocess (QC + normalize + embed)
hsm preprocess
hsm preprocess --dataset heart_cellxgene --force       # re-run with force

# 3. Score gene programs
hsm score
hsm score --program fibrosis --program hypertrophy     # specific programs

# 4. Secretome analysis
hsm secretome
hsm secretome --min-expr-fraction 0.1

# 5. Cross-tissue overlap
hsm overlap
hsm overlap --tissue liver --tissue skeletal_muscle

# 6. Identify systemic responders (requires responder datasets)
hsm responders --min-correlation 0.4

# 7. Rank candidates
hsm rank --top-n 30 --output-format xlsx

# 8. Generate report
hsm report --format html --open
```

### Resume from a checkpoint

```bash
# Skip download and preprocess, start from scoring
hsm run-all --start-from score --config my_config.yaml

# Skip the report step
hsm run-all --skip report
```

### Verbose / debug mode

```bash
hsm preprocess --verbose
hsm score --verbose --dataset heart_cellxgene
```

---

## Manual Data Placement

Some datasets require manual download due to access restrictions or large
file sizes.  Place files as follows before running the pipeline:

### GEO Datasets

Download from https://www.ncbi.nlm.nih.gov/geo/ and place files at:

```
data/raw/
├── GSE183852/
│   ├── GSE183852_matrix.mtx.gz        (or .h5ad)
│   ├── GSE183852_barcodes.tsv.gz
│   ├── GSE183852_features.tsv.gz
│   └── GSE183852_metadata.csv
├── GSE135805/
│   └── (same structure)
└── GSE290577/
    └── (same structure)
```

The pipeline auto-detects whether files are in 10x MTX format or h5ad.

### PRIDE Proteomics Datasets

Download from https://www.ebi.ac.uk/pride/ and place files at:

```
data/raw/
├── PXD021371/
│   └── *.txt    (MaxQuant/DIA-NN output tables)
├── PXD059929/
│   └── *.txt
└── PXD060680/
    └── *.txt
```

Expected proteomics file columns (configurable per dataset):
- `Sample` — sample identifier
- `Protein` — protein accession or gene symbol
- `Intensity` — quantification value (LFQ or raw)

### ArrayExpress Bulk RNA-seq (E-MTAB-15659)

Download from https://www.ebi.ac.uk/arrayexpress/ and place at:

```
data/raw/E-MTAB-15659/
├── counts_matrix.tsv     (genes x samples, tab-separated)
└── sample_metadata.tsv   (sample annotations)
```

### CELLxGENE Census (auto-download)

The `heart_cellxgene` dataset is fetched automatically via the
CELLxGENE Census API (requires `pip install -e ".[census]"`).
Set `local_fallback` in config if you have a pre-downloaded h5ad.

### Reference Files

Place these in `data/reference/`:

```
data/reference/
├── lr_database.tsv            (ligand-receptor pairs: ligand, receptor, source)
├── signal_peptide_genes.txt   (one gene symbol per line)
└── secreted_proteins.txt      (UniProt/HPA secreted protein list)
```

Pre-built reference files can be generated with:

```bash
# Download CellChat LR database (requires R + CellChat, or use precomputed)
python scripts/build_lr_database.py --output data/reference/lr_database.tsv

# Download HPA secreted protein list
python scripts/build_secreted_db.py --output data/reference/secreted_proteins.txt
```

---

## Pipeline Steps

### 1. `download`

- Queries GEO FTP for MTX/h5ad files.
- Queries PRIDE REST API for proteomics files.
- Uses CELLxGENE Census Python API for heart atlas data.
- Skips datasets whose `local_path` already contains files.

### 2. `preprocess`

For scRNA-seq / snRNA-seq:
1. Load raw counts (10x MTX, h5ad, or loom).
2. QC: filter by `min_genes`, `max_genes`, `max_mito_fraction`.
3. Doublet detection (scrublet, optional).
4. Normalize to `target_sum`, log1p transform.
5. Select `n_top_genes` highly variable genes.
6. PCA (30 components), Harmony batch correction (`batch_key`).
7. k-NN graph, UMAP embedding.
8. Leiden clustering at `leiden_resolution`.
9. Save to `processed_dir/<dataset_name>.h5ad`.

For proteomics:
1. Load MaxQuant/DIA-NN output tables.
2. Log2 transform, missing value imputation.
3. Batch correction (if multiple runs).
4. Save to `processed_dir/<dataset_name>_proteomics.h5ad`.

### 3. `score`

- Calls `scanpy.tl.score_genes()` for each program in `config.scoring`.
- Adds score as `.obs["{program}_score"]` to each processed AnnData.
- Generates score distribution plots per cell type.

### 4. `secretome`

- Filters genes expressed in >= `min_expr_fraction` of cardiomyocytes/fibroblasts.
- Intersects with signal peptide prediction database.
- Intersects with detected proteins in PXD059929 conditioned-media dataset.
- Outputs ranked candidate list with multi-evidence scores.

### 5. `overlap`

- Runs pseudo-bulk differential expression per tissue.
- Computes Jaccard index and Fisher's exact test for gene set overlaps.
- Identifies shared up/down regulated genes across heart + systemic tissues.

### 6. `responders`

- Loads enabled responder datasets (liver, muscle, custom).
- Correlates cardiac fibrosis/hypertrophy scores with per-cell expression
  signatures in each responder tissue.
- Identifies cell populations with significant correlation (Spearman r >= threshold).

### 7. `rank`

- Aggregates evidence from steps 3–6.
- Computes composite score per candidate gene/protein.
- Applies configurable weights (scrna, proteomics, LR, cross-tissue, literature).
- Outputs ranked table with all sub-scores.

### 8. `report`

- Collects all figures and tables from `results/`.
- Renders an HTML report with Plotly interactive figures.
- Optionally exports to PDF or Jupyter Notebook.

---

## Output Structure

```
results/
├── logs/
│   └── 20240101_120000_pipeline.log
├── figures/
│   ├── umap_heart_cellxgene.html
│   ├── fibrosis_score_violin.html
│   ├── secretome_heatmap.html
│   └── cross_tissue_overlap.html
├── tables/
│   ├── secretome_candidates.tsv
│   ├── cross_tissue_overlap.tsv
│   ├── systemic_responders.tsv
│   └── ranked_candidates.tsv
├── report/
│   └── analysis_report.html
└── pipeline_status.json

data/
├── raw/           (downloaded / manually placed)
├── processed/     (preprocessed h5ad files)
├── cache/         (pickle cache for fast re-loading)
└── reference/     (LR DB, signal peptide DB, secreted protein DB)
```

---

## Troubleshooting

### Out of memory (OOM)

Reduce `preprocessing.downsample_to` in your config:

```yaml
preprocessing:
  downsample_to: 30000
```

Or run with lower `n_top_genes`:

```yaml
preprocessing:
  n_top_genes: 2000
```

### CELLxGENE Census not found

Install the optional dependency:

```bash
pip install -e ".[census]"
```

Or set `heart_cellxgene.enabled: false` and use a local h5ad via
`heart_cellxgene.local_fallback: path/to/file.h5ad`.

### Slow preprocessing

Increase `execution.n_workers` in config (up to your CPU count):

```yaml
execution:
  n_workers: 8
```

Enable caching so repeated runs skip already-processed datasets:

```yaml
execution:
  use_cache: true
```

### GEO / PRIDE download failures

Many large GEO datasets must be downloaded manually via the web UI due to FTP
size limits.  See [Manual Data Placement](#manual-data-placement).

### Missing reference files

Run the reference-building scripts or download pre-built versions:

```bash
python scripts/build_lr_database.py
python scripts/build_secreted_db.py
```

---

## Citation

If you use this pipeline in your research, please cite:

> Brown Biotech. Heart-Systemic Multi-modal Analysis Pipeline (2024).
> https://github.com/brownbiotech/heart-systemic-multimodal

---

## License

MIT License. See `LICENSE` for details.
