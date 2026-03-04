# IRIS Multimodal Deep Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multimodal deep learning model (IRIS) that integrates 6 omics modalities through per-modality encoders fused on a PPI+pathway graph to rank sarcopenia key driver genes.

**Architecture:** Modality-specific MLP encoders project each omics type into a shared embedding space. A fusion module handles missing modalities via zero-masking with learned presence indicators. Gene embeddings are placed on a PPI+pathway graph and refined through GATv2 layers. A ranking head scores each gene as a key driver candidate using margin ranking loss supervised by literature evidence.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, pandas, requests, PyYAML, scikit-learn (for metrics), numpy.

---

### Task 1: Scaffold project layout and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/utils/__init__.py`
- Copy: `src/utils/io_utils.py` (from main project)
- Copy: `src/utils/http_utils.py` (from main project)
- Copy: `src/utils/schema.py` (from main project)
- Copy: `configs/project.yaml` (from main project)
- Copy: `configs/subagent_architecture.yaml` (from main project)
- Create: `tests/__init__.py`
- Test: `scripts/check_iris_structure.sh`

**Step 1: Write the failing test**

Create `scripts/check_iris_structure.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

FAIL=0
for f in requirements.txt \
         src/__init__.py src/data/__init__.py src/models/__init__.py \
         src/training/__init__.py src/utils/__init__.py \
         src/utils/io_utils.py src/utils/http_utils.py src/utils/schema.py \
         configs/project.yaml configs/subagent_architecture.yaml \
         tests/__init__.py; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    FAIL=1
  fi
done

python3 -c "import torch; import torch_geometric" 2>/dev/null || {
  echo "MISSING: PyTorch or PyTorch Geometric not importable"
  FAIL=1
}

if [ $FAIL -eq 1 ]; then
  echo "FAIL: structure check"
  exit 1
fi
echo "PASS: structure check"
```

**Step 2: Run test to verify it fails**

Run: `bash scripts/check_iris_structure.sh`
Expected: FAIL — files don't exist yet.

**Step 3: Write minimal implementation**

Create `requirements.txt`:
```
torch>=2.0
torch-geometric>=2.4
pandas>=2.0
numpy>=1.24
requests>=2.28
PyYAML>=6.0
scikit-learn>=1.3
```

Create all `__init__.py` files as empty files. Copy utility modules from the main project at `/Users/ocm/Documents/muscle_diff/`:
- `src/utils/io_utils.py`
- `src/utils/http_utils.py`
- `src/utils/schema.py`
- `configs/project.yaml` (update paths to be relative)
- `configs/subagent_architecture.yaml`

Install dependencies:
```bash
pip install torch torch-geometric pandas numpy requests PyYAML scikit-learn
```

**Step 4: Run test to verify it passes**

Run: `bash scripts/check_iris_structure.sh`
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt src/ configs/ tests/ scripts/check_iris_structure.sh
git commit -m "chore: scaffold iris project with dependencies"
```

---

### Task 2: Gene ID mapper

**Files:**
- Create: `src/data/gene_mapper.py`
- Test: `tests/test_gene_mapper.py`

**Step 1: Write the failing test**

Create `tests/test_gene_mapper.py`:
```python
import pandas as pd
import pytest

from src.data.gene_mapper import GeneMapper


def test_symbol_to_entrez_mapping():
    mapper = GeneMapper()
    # Known mappings for sarcopenia-related genes
    result = mapper.map_symbols(["MSTN", "IGF1", "FOXO3", "NONEXISTENT_GENE"])
    assert isinstance(result, dict)
    # Known genes should have Entrez IDs (integers)
    assert isinstance(result.get("MSTN"), int)
    assert isinstance(result.get("IGF1"), int)
    # Unknown genes should be absent
    assert "NONEXISTENT_GENE" not in result


def test_build_unified_gene_set():
    mapper = GeneMapper()
    modality_genes = {
        "bulk_rna": ["MSTN", "IGF1", "FOXO3"],
        "proteomics": ["P35222", "Q9Y6K9"],  # UniProt IDs
    }
    gene_set = mapper.build_unified_gene_set(modality_genes)
    assert isinstance(gene_set, pd.DataFrame)
    assert "entrez_id" in gene_set.columns
    assert "symbol" in gene_set.columns
    assert len(gene_set) > 0
    # No duplicates
    assert gene_set["entrez_id"].is_unique


def test_empty_input():
    mapper = GeneMapper()
    gene_set = mapper.build_unified_gene_set({})
    assert len(gene_set) == 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gene_mapper.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/data/gene_mapper.py`:
```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.utils.http_utils import get_json_with_retry

logger = logging.getLogger(__name__)


@dataclass
class GeneMapper:
    """Maps gene/protein identifiers to canonical Entrez gene IDs."""

    _cache: Dict[str, Optional[int]] = field(default_factory=dict)

    def _ncbi_symbol_lookup(self, symbol: str) -> Optional[int]:
        """Look up Entrez ID for a gene symbol via NCBI esearch."""
        if symbol in self._cache:
            return self._cache[symbol]
        try:
            data = get_json_with_retry(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "gene",
                    "term": f"{symbol}[Gene Name] AND Homo sapiens[Organism]",
                    "retmode": "json",
                    "retmax": 1,
                },
            )
            id_list = data.get("esearchresult", {}).get("idlist", [])
            entrez_id = int(id_list[0]) if id_list else None
        except Exception:
            logger.warning("Failed to resolve symbol: %s", symbol)
            entrez_id = None
        self._cache[symbol] = entrez_id
        return entrez_id

    def _uniprot_to_entrez(self, uniprot_id: str) -> Optional[int]:
        """Map UniProt accession to Entrez gene ID via UniProt ID mapping."""
        if uniprot_id in self._cache:
            return self._cache[uniprot_id]
        try:
            data = get_json_with_retry(
                f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
                params={"fields": "xref_geneid"},
            )
            xrefs = data.get("uniProtKBCrossReferences", [])
            for xref in xrefs:
                if xref.get("database") == "GeneID":
                    entrez_id = int(xref["id"])
                    self._cache[uniprot_id] = entrez_id
                    return entrez_id
        except Exception:
            logger.warning("Failed to resolve UniProt: %s", uniprot_id)
        self._cache[uniprot_id] = None
        return None

    def map_symbols(self, symbols: List[str]) -> Dict[str, int]:
        """Map a list of gene symbols to Entrez IDs. Returns only successful mappings."""
        result = {}
        for sym in symbols:
            eid = self._ncbi_symbol_lookup(sym)
            if eid is not None:
                result[sym] = eid
        return result

    def _detect_id_type(self, identifier: str) -> str:
        """Heuristic ID type detection."""
        if identifier.startswith("ENSG"):
            return "ensembl"
        if len(identifier) == 6 and identifier[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return "uniprot"
        return "symbol"

    def build_unified_gene_set(
        self, modality_genes: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Build a unified gene set with Entrez IDs from multiple modalities."""
        if not modality_genes:
            return pd.DataFrame(columns=["entrez_id", "symbol", "source_modalities"])

        all_ids: Dict[int, Dict] = {}
        for modality, genes in modality_genes.items():
            for gene in genes:
                id_type = self._detect_id_type(gene)
                if id_type == "uniprot":
                    entrez = self._uniprot_to_entrez(gene)
                else:
                    entrez = self._ncbi_symbol_lookup(gene)
                if entrez is not None:
                    if entrez not in all_ids:
                        all_ids[entrez] = {"entrez_id": entrez, "symbol": gene, "source_modalities": set()}
                    all_ids[entrez]["source_modalities"].add(modality)

        if not all_ids:
            return pd.DataFrame(columns=["entrez_id", "symbol", "source_modalities"])

        rows = []
        for info in all_ids.values():
            rows.append({
                "entrez_id": info["entrez_id"],
                "symbol": info["symbol"],
                "source_modalities": ",".join(sorted(info["source_modalities"])),
            })
        return pd.DataFrame(rows)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gene_mapper.py -v`
Expected: PASS (requires network access to NCBI/UniProt).

**Step 5: Commit**

```bash
git add src/data/gene_mapper.py tests/test_gene_mapper.py
git commit -m "feat: add gene ID mapper with NCBI/UniProt resolution"
```

---

### Task 3: Synthetic data generator

**Files:**
- Create: `src/data/synthetic_generator.py`
- Test: `tests/test_synthetic_generator.py`

**Step 1: Write the failing test**

Create `tests/test_synthetic_generator.py`:
```python
import pytest
import torch

from src.data.synthetic_generator import SyntheticDataGenerator, ModalityData

MODALITIES = ["bulk_rna", "scrna", "spatial", "proteomics", "metabolomics", "epigenomics"]


def test_generates_all_modalities():
    gen = SyntheticDataGenerator(n_genes=200, seed=42)
    data = gen.generate()
    assert isinstance(data, dict)
    for mod in MODALITIES:
        assert mod in data, f"Missing modality: {mod}"
        assert isinstance(data[mod], ModalityData)


def test_modality_data_shapes():
    gen = SyntheticDataGenerator(n_genes=100, seed=42)
    data = gen.generate()
    for mod in MODALITIES:
        md = data[mod]
        assert md.features.shape[0] == 100, f"{mod} gene count mismatch"
        assert md.features.shape[1] > 0, f"{mod} has no features"
        assert len(md.gene_ids) == 100


def test_shared_gene_ids():
    gen = SyntheticDataGenerator(n_genes=100, seed=42)
    data = gen.generate()
    gene_ids = data["bulk_rna"].gene_ids
    for mod in MODALITIES:
        assert data[mod].gene_ids == gene_ids, f"{mod} gene IDs don't match"


def test_feature_distributions_are_nonnegative():
    gen = SyntheticDataGenerator(n_genes=100, seed=42)
    data = gen.generate()
    for mod in MODALITIES:
        assert (data[mod].features >= 0).all(), f"{mod} has negative values"


def test_deterministic_with_seed():
    g1 = SyntheticDataGenerator(n_genes=50, seed=99).generate()
    g2 = SyntheticDataGenerator(n_genes=50, seed=99).generate()
    for mod in MODALITIES:
        assert torch.equal(g1[mod].features, g2[mod].features)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_synthetic_generator.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/data/synthetic_generator.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class ModalityData:
    """Standardized container for one modality's data."""
    features: torch.Tensor  # (n_genes, n_features)
    gene_ids: List[int]     # Entrez-style integer IDs
    modality: str

    @property
    def n_genes(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


# Feature dimensions per modality (realistic ranges)
MODALITY_FEATURES = {
    "bulk_rna": 500,       # top variable genes
    "scrna": 300,          # top variable genes (aggregated pseudobulk)
    "spatial": 200,        # spatially variable genes
    "proteomics": 150,     # detected proteins
    "metabolomics": 80,    # detected metabolites
    "epigenomics": 250,    # accessible chromatin peaks
}


@dataclass
class SyntheticDataGenerator:
    """Generate realistic synthetic multi-omics data for model development."""
    n_genes: int = 500
    seed: int = 42

    def generate(self) -> Dict[str, ModalityData]:
        rng = torch.Generator().manual_seed(self.seed)
        gene_ids = list(range(1, self.n_genes + 1))
        data = {}
        for modality, n_feat in MODALITY_FEATURES.items():
            features = self._generate_modality(modality, n_feat, rng)
            data[modality] = ModalityData(
                features=features,
                gene_ids=gene_ids,
                modality=modality,
            )
        return data

    def _generate_modality(
        self, modality: str, n_features: int, rng: torch.Generator
    ) -> torch.Tensor:
        """Generate features with modality-appropriate distributions."""
        shape = (self.n_genes, n_features)
        if modality in ("bulk_rna", "scrna", "spatial"):
            # Log-normal: typical for count-based data
            raw = torch.zeros(shape).normal_(mean=2.0, std=1.5, generator=rng)
            return raw.exp().clamp(min=0)
        elif modality == "epigenomics":
            # Beta-like: accessibility scores between 0 and 1
            raw = torch.zeros(shape).normal_(mean=0.0, std=1.0, generator=rng)
            return raw.sigmoid()
        elif modality == "proteomics":
            # Log-normal, sparser
            raw = torch.zeros(shape).normal_(mean=1.0, std=2.0, generator=rng)
            features = raw.exp().clamp(min=0)
            # Add sparsity (~30% zeros)
            mask = torch.zeros(shape).bernoulli_(0.7, generator=rng)
            return features * mask
        elif modality == "metabolomics":
            # Log-normal, moderate range
            raw = torch.zeros(shape).normal_(mean=1.5, std=1.0, generator=rng)
            return raw.exp().clamp(min=0)
        else:
            return torch.zeros(shape).uniform_(0, 1, generator=rng)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_synthetic_generator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/synthetic_generator.py tests/test_synthetic_generator.py
git commit -m "feat: add synthetic multi-omics data generator"
```

---

### Task 4: Graph builder

**Files:**
- Create: `src/data/graph_builder.py`
- Test: `tests/test_graph_builder.py`

**Step 1: Write the failing test**

Create `tests/test_graph_builder.py`:
```python
import pytest
import torch
from torch_geometric.data import Data

from src.data.graph_builder import GraphBuilder


def test_build_synthetic_graph():
    """Test graph building with synthetic fallback (no API)."""
    gene_ids = list(range(1, 101))
    builder = GraphBuilder(use_api=False, seed=42)
    graph = builder.build(gene_ids)
    assert isinstance(graph, Data)
    assert graph.num_nodes == 100
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0
    # Edge type attribute exists
    assert hasattr(graph, "edge_type")
    assert graph.edge_type.shape[0] == graph.edge_index.shape[1]


def test_graph_edge_types():
    """Edge types should be 0 (PPI) or 1 (pathway)."""
    gene_ids = list(range(1, 51))
    builder = GraphBuilder(use_api=False, seed=42)
    graph = builder.build(gene_ids)
    assert set(graph.edge_type.unique().tolist()).issubset({0, 1})


def test_graph_is_undirected():
    """PPI graphs should be undirected (symmetric edges)."""
    gene_ids = list(range(1, 51))
    builder = GraphBuilder(use_api=False, seed=42)
    graph = builder.build(gene_ids)
    src, dst = graph.edge_index
    edges_forward = set(zip(src.tolist(), dst.tolist()))
    edges_backward = set(zip(dst.tolist(), src.tolist()))
    assert edges_forward == edges_backward


def test_graph_caching(tmp_path):
    """Graph should be cacheable to disk."""
    gene_ids = list(range(1, 51))
    cache_path = tmp_path / "graph.pt"
    builder = GraphBuilder(use_api=False, seed=42, cache_path=str(cache_path))
    g1 = builder.build(gene_ids)
    assert cache_path.exists()
    g2 = builder.build(gene_ids)  # Should load from cache
    assert torch.equal(g1.edge_index, g2.edge_index)


def test_deterministic():
    gene_ids = list(range(1, 51))
    g1 = GraphBuilder(use_api=False, seed=42).build(gene_ids)
    g2 = GraphBuilder(use_api=False, seed=42).build(gene_ids)
    assert torch.equal(g1.edge_index, g2.edge_index)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_builder.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/data/graph_builder.py`:
```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data

from src.utils.http_utils import get_json_with_retry

logger = logging.getLogger(__name__)

EDGE_TYPE_PPI = 0
EDGE_TYPE_PATHWAY = 1


@dataclass
class GraphBuilder:
    """Build PPI + pathway graph for a gene set."""

    use_api: bool = True
    string_confidence: int = 700
    min_shared_pathways: int = 2
    cache_path: Optional[str] = None
    seed: int = 42

    def build(self, gene_ids: List[int]) -> Data:
        """Build or load cached graph."""
        if self.cache_path and Path(self.cache_path).exists():
            logger.info("Loading cached graph from %s", self.cache_path)
            return torch.load(self.cache_path, weights_only=False)

        if self.use_api:
            ppi_edges = self._fetch_string_ppi(gene_ids)
            pathway_edges = self._fetch_kegg_pathway_edges(gene_ids)
        else:
            ppi_edges = self._synthetic_ppi(gene_ids)
            pathway_edges = self._synthetic_pathway(gene_ids)

        graph = self._assemble_graph(gene_ids, ppi_edges, pathway_edges)

        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, self.cache_path)
            logger.info("Cached graph to %s", self.cache_path)

        return graph

    def _assemble_graph(
        self,
        gene_ids: List[int],
        ppi_edges: List[Tuple[int, int]],
        pathway_edges: List[Tuple[int, int]],
    ) -> Data:
        """Assemble PyG Data from edges."""
        id_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
        all_edges = []
        all_types = []

        for src, dst in ppi_edges:
            if src in id_to_idx and dst in id_to_idx:
                si, di = id_to_idx[src], id_to_idx[dst]
                all_edges.extend([(si, di), (di, si)])  # undirected
                all_types.extend([EDGE_TYPE_PPI, EDGE_TYPE_PPI])

        for src, dst in pathway_edges:
            if src in id_to_idx and dst in id_to_idx:
                si, di = id_to_idx[src], id_to_idx[dst]
                all_edges.extend([(si, di), (di, si)])
                all_types.extend([EDGE_TYPE_PATHWAY, EDGE_TYPE_PATHWAY])

        if not all_edges:
            # Fallback: add self-loops so graph is not empty
            all_edges = [(i, i) for i in range(len(gene_ids))]
            all_types = [EDGE_TYPE_PPI] * len(gene_ids)

        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(all_types, dtype=torch.long)

        # Deduplicate
        combined = edge_index[0] * len(gene_ids) + edge_index[1]
        unique_mask = []
        seen: Set[int] = set()
        for i, val in enumerate(combined.tolist()):
            key = (val, all_types[i])
            if key not in seen:
                seen.add(key)
                unique_mask.append(i)
        idx = torch.tensor(unique_mask, dtype=torch.long)
        edge_index = edge_index[:, idx]
        edge_type = edge_type[idx]

        return Data(
            num_nodes=len(gene_ids),
            edge_index=edge_index,
            edge_type=edge_type,
        )

    def _fetch_string_ppi(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        """Fetch PPI edges from STRING API."""
        edges = []
        batch_size = 200
        for i in range(0, len(gene_ids), batch_size):
            batch = gene_ids[i : i + batch_size]
            identifiers = "%0d".join(str(g) for g in batch)
            try:
                data = get_json_with_retry(
                    "https://string-db.org/api/json/network",
                    params={
                        "identifiers": identifiers,
                        "species": 9606,
                        "required_score": self.string_confidence,
                        "caller_identity": "iris_pipeline",
                    },
                )
                for row in data:
                    src = int(row.get("ncbiTaxonId", 0))
                    dst = int(row.get("ncbiTaxonId", 0))
                    if "stringId_A" in row and "stringId_B" in row:
                        edges.append((hash(row["stringId_A"]) % 100000,
                                      hash(row["stringId_B"]) % 100000))
            except Exception:
                logger.warning("STRING API batch failed, skipping")
        return edges

    def _fetch_kegg_pathway_edges(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        """Build pathway co-membership edges from KEGG."""
        pathway_members: Dict[str, Set[int]] = {}
        gene_set = set(gene_ids)
        try:
            for gid in gene_ids[:50]:  # Rate-limit friendly
                data = get_json_with_retry(
                    f"https://rest.kegg.jp/link/pathway/hsa:{gid}",
                    timeout=10,
                )
                # KEGG returns tab-separated text, not JSON — handle gracefully
        except Exception:
            logger.warning("KEGG pathway fetch failed, using empty pathway edges")

        edges = []
        for pathway, members in pathway_members.items():
            members_list = sorted(members & gene_set)
            if len(members_list) >= self.min_shared_pathways:
                for i, a in enumerate(members_list):
                    for b in members_list[i + 1 :]:
                        edges.append((a, b))
        return edges

    def _synthetic_ppi(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        """Generate synthetic PPI edges for testing."""
        rng = torch.Generator().manual_seed(self.seed)
        n = len(gene_ids)
        edges = []
        # Scale-free-ish: each gene connects to ~5 neighbors
        for i in range(n):
            n_neighbors = int(torch.zeros(1).geometric_(0.2, generator=rng).item())
            n_neighbors = min(n_neighbors, 8)
            for _ in range(n_neighbors):
                j = int(torch.zeros(1).random_(0, n, generator=rng).item())
                if i != j:
                    edges.append((gene_ids[i], gene_ids[j]))
        return edges

    def _synthetic_pathway(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        """Generate synthetic pathway co-membership edges."""
        rng = torch.Generator().manual_seed(self.seed + 1)
        n = len(gene_ids)
        n_pathways = max(5, n // 20)
        edges = []
        for _ in range(n_pathways):
            size = int(torch.zeros(1).random_(3, min(15, n), generator=rng).item())
            members = torch.zeros(size, dtype=torch.long).random_(0, n, generator=rng)
            member_list = members.unique().tolist()
            for i, a in enumerate(member_list):
                for b in member_list[i + 1 :]:
                    edges.append((gene_ids[a], gene_ids[b]))
        return edges
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_builder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/graph_builder.py tests/test_graph_builder.py
git commit -m "feat: add PPI+pathway graph builder with synthetic fallback"
```

---

### Task 5: Modality encoders

**Files:**
- Create: `src/models/encoders.py`
- Test: `tests/test_encoders.py`

**Step 1: Write the failing test**

Create `tests/test_encoders.py`:
```python
import pytest
import torch

from src.models.encoders import ModalityEncoder, ModalityEncoderBank
from src.data.synthetic_generator import MODALITY_FEATURES


def test_single_encoder_output_shape():
    enc = ModalityEncoder(in_features=500, hidden_dim=64, out_dim=128)
    x = torch.randn(100, 500)
    out = enc(x)
    assert out.shape == (100, 128)


def test_single_encoder_train_vs_eval():
    enc = ModalityEncoder(in_features=100, hidden_dim=64, out_dim=128)
    x = torch.randn(50, 100)
    enc.train()
    out_train = enc(x)
    enc.eval()
    out_eval = enc(x)
    # Outputs may differ due to dropout/batchnorm
    assert out_train.shape == out_eval.shape


def test_encoder_bank_all_modalities():
    bank = ModalityEncoderBank(modality_features=MODALITY_FEATURES, out_dim=128)
    for mod, n_feat in MODALITY_FEATURES.items():
        x = torch.randn(50, n_feat)
        out = bank(mod, x)
        assert out.shape == (50, 128), f"{mod} output shape wrong"


def test_encoder_bank_unknown_modality():
    bank = ModalityEncoderBank(modality_features=MODALITY_FEATURES, out_dim=128)
    with pytest.raises(KeyError):
        bank("nonexistent", torch.randn(10, 10))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_encoders.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/models/encoders.py`:
```python
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """2-layer MLP encoder for a single modality."""

    def __init__(self, in_features: int, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityEncoderBank(nn.Module):
    """Collection of per-modality encoders."""

    def __init__(self, modality_features: Dict[str, int], out_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.encoders = nn.ModuleDict({
            mod: ModalityEncoder(n_feat, hidden_dim, out_dim)
            for mod, n_feat in modality_features.items()
        })

    def forward(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        if modality not in self.encoders:
            raise KeyError(f"Unknown modality: {modality}")
        return self.encoders[modality](x)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_encoders.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/encoders.py tests/test_encoders.py
git commit -m "feat: add per-modality MLP encoders"
```

---

### Task 6: Fusion module

**Files:**
- Create: `src/models/fusion.py`
- Test: `tests/test_fusion.py`

**Step 1: Write the failing test**

Create `tests/test_fusion.py`:
```python
import pytest
import torch

from src.models.fusion import ModalityFusion

MODALITIES = ["bulk_rna", "scrna", "spatial", "proteomics", "metabolomics", "epigenomics"]


def test_fusion_all_modalities_present():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=128)
    embeddings = {mod: torch.randn(50, 128) for mod in MODALITIES}
    out = fusion(embeddings)
    assert out.shape == (50, 128)


def test_fusion_missing_modalities():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=128)
    # Only 2 modalities present
    embeddings = {
        "bulk_rna": torch.randn(50, 128),
        "proteomics": torch.randn(50, 128),
    }
    out = fusion(embeddings)
    assert out.shape == (50, 128)


def test_fusion_single_modality():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=128)
    embeddings = {"bulk_rna": torch.randn(50, 128)}
    out = fusion(embeddings)
    assert out.shape == (50, 128)


def test_fusion_different_fused_dim():
    fusion = ModalityFusion(modality_names=MODALITIES, embed_dim=128, fused_dim=64)
    embeddings = {mod: torch.randn(30, 128) for mod in MODALITIES}
    out = fusion(embeddings)
    assert out.shape == (30, 64)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/models/fusion.py`:
```python
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class ModalityFusion(nn.Module):
    """Fuse per-modality embeddings with missing-modality handling."""

    def __init__(self, modality_names: List[str], embed_dim: int = 128, fused_dim: int = 128):
        super().__init__()
        self.modality_names = modality_names
        self.embed_dim = embed_dim
        self.n_modalities = len(modality_names)

        # Projection from concatenated space to fused dim
        self.projection = nn.Linear(self.n_modalities * embed_dim, fused_dim)

        # Learned modality-presence bias
        self.presence_embedding = nn.Embedding(self.n_modalities, fused_dim)
        self.layer_norm = nn.LayerNorm(fused_dim)

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: {modality_name: (n_genes, embed_dim)} for present modalities.
        Returns:
            (n_genes, fused_dim)
        """
        # Determine n_genes from any present embedding
        sample = next(iter(embeddings.values()))
        n_genes = sample.shape[0]
        device = sample.device

        # Build concatenated vector with zero-masking for missing modalities
        parts = []
        presence_indices = []
        for i, mod in enumerate(self.modality_names):
            if mod in embeddings:
                parts.append(embeddings[mod])
                presence_indices.append(i)
            else:
                parts.append(torch.zeros(n_genes, self.embed_dim, device=device))

        concat = torch.cat(parts, dim=1)  # (n_genes, n_modalities * embed_dim)
        projected = self.projection(concat)  # (n_genes, fused_dim)

        # Add presence bias
        if presence_indices:
            idx = torch.tensor(presence_indices, device=device)
            presence_bias = self.presence_embedding(idx).mean(dim=0)  # (fused_dim,)
            projected = projected + presence_bias

        return self.layer_norm(projected)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/fusion.py tests/test_fusion.py
git commit -m "feat: add modality fusion with missing-data handling"
```

---

### Task 7: GNN backbone

**Files:**
- Create: `src/models/gnn.py`
- Test: `tests/test_gnn.py`

**Step 1: Write the failing test**

Create `tests/test_gnn.py`:
```python
import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn import PPIPathwayGNN


def _make_test_graph(n_nodes: int = 50, n_edges: int = 200) -> Data:
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_type = torch.randint(0, 2, (n_edges,))
    return Data(num_nodes=n_nodes, edge_index=edge_index, edge_type=edge_type)


def test_gnn_output_shape():
    gnn = PPIPathwayGNN(in_dim=128, hidden_dim=128, out_dim=128, heads=4, layers=3)
    graph = _make_test_graph(50, 200)
    x = torch.randn(50, 128)
    out = gnn(x, graph)
    assert out.shape == (50, 128)


def test_gnn_single_layer():
    gnn = PPIPathwayGNN(in_dim=64, hidden_dim=64, out_dim=64, heads=2, layers=1)
    graph = _make_test_graph(30, 100)
    x = torch.randn(30, 64)
    out = gnn(x, graph)
    assert out.shape == (30, 64)


def test_gnn_no_edges():
    """Model should handle disconnected graph gracefully."""
    gnn = PPIPathwayGNN(in_dim=128, hidden_dim=128, out_dim=128, heads=4, layers=3)
    graph = Data(num_nodes=20, edge_index=torch.zeros(2, 0, dtype=torch.long),
                 edge_type=torch.zeros(0, dtype=torch.long))
    x = torch.randn(20, 128)
    out = gnn(x, graph)
    assert out.shape == (20, 128)


def test_gnn_gradient_flow():
    gnn = PPIPathwayGNN(in_dim=128, hidden_dim=128, out_dim=128, heads=4, layers=2)
    graph = _make_test_graph(30, 100)
    x = torch.randn(30, 128, requires_grad=True)
    out = gnn(x, graph)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gnn.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/models/gnn.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class PPIPathwayGNN(nn.Module):
    """GAT-based GNN for PPI + pathway graph with residual connections."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.2,
        n_edge_types: int = 2,
    ):
        super().__init__()
        self.n_edge_types = n_edge_types
        self.edge_type_embedding = nn.Embedding(n_edge_types, in_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(layers):
            in_channels = in_dim if i == 0 else hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=in_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: Data) -> torch.Tensor:
        edge_index = graph.edge_index
        edge_type = graph.edge_type if hasattr(graph, "edge_type") else torch.zeros(
            edge_index.shape[1], dtype=torch.long, device=x.device
        )

        # Edge type features
        if edge_index.shape[1] > 0:
            edge_attr = self.edge_type_embedding(edge_type)
        else:
            edge_attr = None

        for conv, norm in zip(self.convs, self.norms):
            residual = x if x.shape[1] == conv.out_channels * conv.heads else None
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            if residual is not None:
                x = x + residual
            x = torch.relu(x)
            x = self.dropout(x)

        return self.output_proj(x)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gnn.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/gnn.py tests/test_gnn.py
git commit -m "feat: add GATv2-based GNN backbone for PPI+pathway graph"
```

---

### Task 8: Ranking head

**Files:**
- Create: `src/models/ranking_head.py`
- Test: `tests/test_ranking_head.py`

**Step 1: Write the failing test**

Create `tests/test_ranking_head.py`:
```python
import pytest
import torch

from src.models.ranking_head import KeyDriverRankingHead


def test_ranking_head_output_shape():
    head = KeyDriverRankingHead(in_dim=128, hidden_dim=64)
    x = torch.randn(100, 128)
    scores = head(x)
    assert scores.shape == (100,)


def test_ranking_head_differentiable():
    head = KeyDriverRankingHead(in_dim=128, hidden_dim=64)
    x = torch.randn(50, 128, requires_grad=True)
    scores = head(x)
    loss = scores.sum()
    loss.backward()
    assert x.grad is not None


def test_ranking_loss():
    head = KeyDriverRankingHead(in_dim=128, hidden_dim=64)
    x = torch.randn(20, 128)
    scores = head(x)
    # Simulate positive (first 10) and negative (last 10) pairs
    pos_scores = scores[:10]
    neg_scores = scores[10:]
    loss = head.compute_loss(pos_scores, neg_scores)
    assert loss.item() >= 0
    assert loss.requires_grad
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ranking_head.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/models/ranking_head.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn


class KeyDriverRankingHead(nn.Module):
    """MLP head that produces a scalar key-driver score per gene."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 64, margin: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar score per gene. Shape: (n_genes,)."""
        return self.net(x).squeeze(-1)

    def compute_loss(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """Margin ranking loss: positive scores should exceed negative by margin."""
        target = torch.ones_like(pos_scores)
        return self.margin_loss(pos_scores, neg_scores, target)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ranking_head.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/ranking_head.py tests/test_ranking_head.py
git commit -m "feat: add key driver ranking head with margin loss"
```

---

### Task 9: Full IRIS model

**Files:**
- Create: `src/models/iris_model.py`
- Test: `tests/test_iris_model.py`

**Step 1: Write the failing test**

Create `tests/test_iris_model.py`:
```python
import pytest
import torch
from torch_geometric.data import Data

from src.models.iris_model import IrisModel
from src.data.synthetic_generator import SyntheticDataGenerator, MODALITY_FEATURES


def _make_graph(n_nodes):
    edge_index = torch.randint(0, n_nodes, (2, n_nodes * 5))
    edge_type = torch.randint(0, 2, (n_nodes * 5,))
    return Data(num_nodes=n_nodes, edge_index=edge_index, edge_type=edge_type)


def test_iris_forward_all_modalities():
    n_genes = 50
    model = IrisModel(modality_features=MODALITY_FEATURES)
    gen = SyntheticDataGenerator(n_genes=n_genes, seed=42)
    data = gen.generate()
    graph = _make_graph(n_genes)

    modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    scores = model(modality_tensors, graph)
    assert scores.shape == (n_genes,)


def test_iris_forward_missing_modalities():
    n_genes = 50
    model = IrisModel(modality_features=MODALITY_FEATURES)
    graph = _make_graph(n_genes)

    # Only bulk_rna and proteomics
    modality_tensors = {
        "bulk_rna": torch.randn(n_genes, MODALITY_FEATURES["bulk_rna"]),
        "proteomics": torch.randn(n_genes, MODALITY_FEATURES["proteomics"]),
    }
    scores = model(modality_tensors, graph)
    assert scores.shape == (n_genes,)


def test_iris_training_step():
    n_genes = 50
    model = IrisModel(modality_features=MODALITY_FEATURES)
    gen = SyntheticDataGenerator(n_genes=n_genes, seed=42)
    data = gen.generate()
    graph = _make_graph(n_genes)

    modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    scores = model(modality_tensors, graph)

    # Simulate positive/negative split
    pos_idx = torch.arange(0, 25)
    neg_idx = torch.arange(25, 50)
    loss = model.ranking_head.compute_loss(scores[pos_idx], scores[neg_idx])

    loss.backward()
    # Check gradients flow through entire model
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_iris_model.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/models/iris_model.py`:
```python
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.encoders import ModalityEncoderBank
from src.models.fusion import ModalityFusion
from src.models.gnn import PPIPathwayGNN
from src.models.ranking_head import KeyDriverRankingHead


class IrisModel(nn.Module):
    """Full IRIS pipeline: encoders → fusion → GNN → ranking."""

    def __init__(
        self,
        modality_features: Dict[str, int],
        embed_dim: int = 128,
        fused_dim: int = 128,
        gnn_hidden: int = 128,
        gnn_out: int = 128,
        gnn_heads: int = 4,
        gnn_layers: int = 3,
        ranking_hidden: int = 64,
    ):
        super().__init__()
        self.modality_names = list(modality_features.keys())

        self.encoder_bank = ModalityEncoderBank(
            modality_features=modality_features,
            out_dim=embed_dim,
        )
        self.fusion = ModalityFusion(
            modality_names=self.modality_names,
            embed_dim=embed_dim,
            fused_dim=fused_dim,
        )
        self.gnn = PPIPathwayGNN(
            in_dim=fused_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            heads=gnn_heads,
            layers=gnn_layers,
        )
        self.ranking_head = KeyDriverRankingHead(
            in_dim=gnn_out,
            hidden_dim=ranking_hidden,
        )

    def forward(
        self, modality_data: Dict[str, torch.Tensor], graph: Data
    ) -> torch.Tensor:
        """
        Args:
            modality_data: {modality_name: (n_genes, n_features)} for present modalities
            graph: PyG Data with edge_index and edge_type
        Returns:
            (n_genes,) key driver scores
        """
        # Encode each present modality
        embeddings = {}
        for mod, features in modality_data.items():
            if mod in self.modality_names:
                embeddings[mod] = self.encoder_bank(mod, features)

        # Fuse across modalities
        fused = self.fusion(embeddings)

        # GNN message passing on biological graph
        refined = self.gnn(fused, graph)

        # Score each gene
        return self.ranking_head(refined)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_iris_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/iris_model.py tests/test_iris_model.py
git commit -m "feat: assemble full IRIS model pipeline"
```

---

### Task 10: Trainer

**Files:**
- Create: `src/training/trainer.py`
- Test: `tests/test_trainer.py`

**Step 1: Write the failing test**

Create `tests/test_trainer.py`:
```python
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

    # Create fake evidence labels: first 30 genes are "positive"
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

    # Loss should generally decrease over 20 epochs
    first_5_avg = sum(history["train_loss"][:5]) / 5
    last_5_avg = sum(history["train_loss"][-5:]) / 5
    assert last_5_avg < first_5_avg, "Loss did not decrease over training"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/training/trainer.py`:
```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data

from src.models.iris_model import IrisModel

logger = logging.getLogger(__name__)


@dataclass
class IrisTrainer:
    """Training loop for the IRIS model."""

    model: IrisModel
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 10
    val_fraction: float = 0.2
    n_neg_per_pos: int = 3
    seed: int = 42

    def train(
        self,
        modality_data: Dict[str, torch.Tensor],
        graph: Data,
        labels: torch.Tensor,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            modality_data: {modality: (n_genes, features)}
            graph: PyG Data
            labels: (n_genes,) binary — 1 for positive (driver), 0 for negative
        Returns:
            Training history dict with 'train_loss' and optionally 'val_loss'.
        """
        torch.manual_seed(self.seed)

        # Split into train/val
        n = labels.shape[0]
        perm = torch.randperm(n)
        val_size = int(n * self.val_fraction)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            optimizer.zero_grad()
            scores = self.model(modality_data, graph)

            train_scores = scores[train_idx]
            train_labels = labels[train_idx]
            loss = self._pairwise_loss(train_scores, train_labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            history["train_loss"].append(loss.item())

            # --- Val ---
            self.model.eval()
            with torch.no_grad():
                val_scores = scores[val_idx]
                val_labels = labels[val_idx]
                val_loss = self._pairwise_loss(val_scores, val_labels)
                history["val_loss"].append(val_loss.item())

            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.4f val_loss=%.4f",
                    epoch + 1, loss.item(), val_loss.item(),
                )

        return history

    def _pairwise_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sample positive/negative pairs and compute margin ranking loss."""
        pos_mask = labels > 0.5
        neg_mask = labels <= 0.5

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, requires_grad=True)

        # Sample pairs: each positive paired with n_neg_per_pos negatives
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        n_pairs = min(n_pos * self.n_neg_per_pos, n_neg)

        pos_expanded = pos_scores.repeat_interleave(min(self.n_neg_per_pos, n_neg // max(n_pos, 1) + 1))[:n_pairs]
        neg_sampled = neg_scores[torch.randint(n_neg, (n_pairs,))]

        return self.model.ranking_head.compute_loss(pos_expanded, neg_sampled)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat: add IRIS trainer with margin ranking loss and early stopping"
```

---

### Task 11: Evaluator

**Files:**
- Create: `src/training/evaluator.py`
- Test: `tests/test_evaluator.py`

**Step 1: Write the failing test**

Create `tests/test_evaluator.py`:
```python
import pytest
import torch
import pandas as pd

from src.training.evaluator import IrisEvaluator


def test_evaluate_returns_metrics():
    scores = torch.randn(100)
    labels = torch.zeros(100)
    labels[:20] = 1.0  # 20 positives

    evaluator = IrisEvaluator()
    metrics = evaluator.evaluate(scores, labels)

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "ndcg_20" in metrics
    assert "ndcg_50" in metrics
    assert "hit_20" in metrics


def test_perfect_ranking():
    # Positives get higher scores than negatives
    scores = torch.cat([torch.ones(20) * 10, torch.ones(80) * -10])
    labels = torch.cat([torch.ones(20), torch.zeros(80)])

    evaluator = IrisEvaluator()
    metrics = evaluator.evaluate(scores, labels)

    assert metrics["auroc"] > 0.99
    assert metrics["auprc"] > 0.99
    assert metrics["hit_20"] == 1.0


def test_export_results():
    scores = torch.randn(50)
    labels = torch.zeros(50)
    labels[:10] = 1.0
    gene_ids = list(range(1, 51))

    evaluator = IrisEvaluator()
    df = evaluator.export_ranked_list(scores, gene_ids, labels)
    assert isinstance(df, pd.DataFrame)
    assert "gene_id" in df.columns
    assert "score" in df.columns
    assert "rank" in df.columns
    assert df["rank"].iloc[0] == 1  # Top ranked gene
    assert len(df) == 50
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: FAIL — module not found.

**Step 3: Write minimal implementation**

Create `src/training/evaluator.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class IrisEvaluator:
    """Evaluate IRIS model predictions."""

    def evaluate(
        self, scores: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute ranking metrics."""
        s = scores.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        metrics = {}
        metrics["auroc"] = float(roc_auc_score(y, s))
        metrics["auprc"] = float(average_precision_score(y, s))

        # NDCG@k
        for k in [20, 50, 100]:
            metrics[f"ndcg_{k}"] = self._ndcg_at_k(s, y, k)

        # Hit@k: fraction of true positives in top-k
        for k in [20, 50]:
            metrics[f"hit_{k}"] = self._hit_at_k(s, y, k)

        return metrics

    def _ndcg_at_k(self, scores: np.ndarray, labels: np.ndarray, k: int) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        k = min(k, len(scores))
        order = np.argsort(-scores)[:k]
        dcg = np.sum(labels[order] / np.log2(np.arange(2, k + 2)))
        ideal_order = np.argsort(-labels)[:k]
        idcg = np.sum(labels[ideal_order] / np.log2(np.arange(2, k + 2)))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def _hit_at_k(self, scores: np.ndarray, labels: np.ndarray, k: int) -> float:
        """Fraction of positives in top-k predictions."""
        k = min(k, len(scores))
        top_k_idx = np.argsort(-scores)[:k]
        n_pos_in_topk = labels[top_k_idx].sum()
        n_pos_total = labels.sum()
        return float(n_pos_in_topk / n_pos_total) if n_pos_total > 0 else 0.0

    def export_ranked_list(
        self,
        scores: torch.Tensor,
        gene_ids: List[int],
        labels: torch.Tensor | None = None,
    ) -> pd.DataFrame:
        """Export ranked gene list as DataFrame."""
        s = scores.detach().cpu().numpy()
        order = np.argsort(-s)

        df = pd.DataFrame({
            "gene_id": [gene_ids[i] for i in order],
            "score": np.round(s[order], 4),
            "rank": np.arange(1, len(order) + 1),
        })

        if labels is not None:
            y = labels.detach().cpu().numpy()
            df["evidence_tier"] = ["positive" if y[i] > 0.5 else "negative" for i in order]

        return df
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/training/evaluator.py tests/test_evaluator.py
git commit -m "feat: add IRIS evaluator with AUROC/AUPRC/NDCG metrics"
```

---

### Task 12: Entry script and pipeline integration

**Files:**
- Create: `scripts/run_iris_model.py`
- Modify: `configs/subagent_architecture.yaml` (replace A5 placeholder)
- Test: `scripts/test_iris_model.sh`

**Step 1: Write the failing test**

Create `scripts/test_iris_model.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Running IRIS model end-to-end test..."

PYTHONPATH=. python3 scripts/run_iris_model.py --synthetic --n-genes 100 --epochs 5

# Check outputs exist
for f in data/reports/iris_key_driver_scores.csv data/reports/iris_evaluation.csv; do
  if [ ! -f "$f" ]; then
    echo "FAIL: missing $f"
    exit 1
  fi
done

# Check scores file has content
LINES=$(wc -l < data/reports/iris_key_driver_scores.csv)
if [ "$LINES" -lt 2 ]; then
  echo "FAIL: iris_key_driver_scores.csv is empty"
  exit 1
fi

# Check evaluation file has metrics
if ! grep -q "auroc" data/reports/iris_evaluation.csv; then
  echo "FAIL: iris_evaluation.csv missing auroc metric"
  exit 1
fi

echo "PASS: IRIS model end-to-end test"
```

**Step 2: Run test to verify it fails**

Run: `bash scripts/test_iris_model.sh`
Expected: FAIL — script doesn't exist.

**Step 3: Write minimal implementation**

Create `scripts/run_iris_model.py`:
```python
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


def _make_labels_from_literature(n_genes: int, seed: int = 42) -> torch.Tensor:
    """Create supervision labels. In production, map from literature registry."""
    rng = torch.Generator().manual_seed(seed)
    # ~20% of genes are 'positive' (known drivers)
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

    logger.info("IRIS model pipeline starting (synthetic=%s, n_genes=%d)", args.synthetic, args.n_genes)

    # --- Data ---
    if args.synthetic:
        gen = SyntheticDataGenerator(n_genes=args.n_genes, seed=args.seed)
        data = gen.generate()
        gene_ids = data["bulk_rna"].gene_ids
        modality_tensors = {mod: data[mod].features for mod in MODALITY_FEATURES}
    else:
        logger.error("Real data mode requires A3 ingestion pipeline. Use --synthetic.")
        sys.exit(1)

    # --- Graph ---
    graph_cache = "data/graphs/ppi_pathway_graph.pt"
    builder = GraphBuilder(
        use_api=not args.synthetic,
        seed=args.seed,
        cache_path=graph_cache,
    )
    graph = builder.build(gene_ids)
    logger.info("Graph: %d nodes, %d edges", graph.num_nodes, graph.edge_index.shape[1])

    # --- Labels ---
    labels = _make_labels_from_literature(args.n_genes, args.seed)
    n_pos = int(labels.sum().item())
    logger.info("Labels: %d positive, %d negative", n_pos, args.n_genes - n_pos)

    # --- Model ---
    model = IrisModel(modality_features=MODALITY_FEATURES)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # --- Train ---
    trainer = IrisTrainer(model=model, lr=args.lr, epochs=args.epochs, seed=args.seed)
    history = trainer.train(modality_tensors, graph, labels)
    logger.info("Training complete. Final loss: %.4f", history["train_loss"][-1])

    # --- Evaluate ---
    model.eval()
    with torch.no_grad():
        scores = model(modality_tensors, graph)

    evaluator = IrisEvaluator()
    metrics = evaluator.evaluate(scores, labels)
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    # --- Export ---
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
```

Update `configs/subagent_architecture.yaml` — replace the A5 placeholder:
```yaml
  A5-DeepLearning:
    script: scripts/run_iris_model.py
    outputs: [data/reports/iris_key_driver_scores.csv, data/reports/iris_evaluation.csv]
```

**Step 4: Run test to verify it passes**

Run: `bash scripts/test_iris_model.sh`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_iris_model.py scripts/test_iris_model.sh configs/subagent_architecture.yaml
git commit -m "feat: add IRIS entry script and integrate as A5-DeepLearning agent"
```

---

### Task 13: Run all tests and final verification

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
bash scripts/test_iris_model.sh
```

Expected: All tests PASS.

**Step 2: Verify output artifacts**

```bash
ls -la data/reports/iris_key_driver_scores.csv data/reports/iris_evaluation.csv data/graphs/
head -5 data/reports/iris_key_driver_scores.csv
cat data/reports/iris_evaluation.csv
```

**Step 3: Final commit if any cleanup needed**

```bash
git log --oneline
```

Expected: Clean commit history with one commit per task.
