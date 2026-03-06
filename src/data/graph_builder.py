from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

EDGE_TYPE_PPI = 0
EDGE_TYPE_PATHWAY = 1

# Major muscle/sarcopenia-related KEGG pathways with core gene members.
# Used as fallback when KEGG API is unavailable.
MUSCLE_PATHWAY_GENES: Dict[str, List[str]] = {
    "mTOR_signaling": [
        "MTOR", "RPTOR", "RICTOR", "RPS6KB1", "EIF4EBP1",
        "AKT1", "AKT2", "PIK3CA", "TSC1", "TSC2", "RHEB",
    ],
    "AMPK_signaling": [
        "PRKAA1", "PRKAA2", "PRKAB1", "PRKAG1",
        "STK11", "CAMKK2", "PPARGC1A", "FOXO3",
    ],
    "autophagy": [
        "ATG5", "ATG7", "ATG12", "BECN1", "BNIP3", "BNIP3L",
        "LAMP2", "CTSL", "ULK1", "ULK2",
    ],
    "ubiquitin_proteasome": [
        "TRIM63", "FBXO32", "FBXO30", "UBR5",
        "PSMA1", "PSMB5", "UBE2D1",
    ],
    "PI3K_Akt": [
        "IGF1", "IGF1R", "IRS1", "IRS2",
        "PIK3CA", "PIK3CB", "PIK3R1",
        "AKT1", "AKT2", "FOXO1", "FOXO3",
    ],
    "TGFbeta_signaling": [
        "MSTN", "TGFB1", "TGFBR1", "TGFBR2",
        "SMAD2", "SMAD3", "SMAD4", "SMAD7",
        "ACVR2B", "GDF11", "BMP7",
    ],
    "myogenesis": [
        "PAX7", "MYOD1", "MYF5", "MYF6", "MYOG",
        "MYH1", "MYH2", "MYH4", "MYH7",
    ],
}


@dataclass
class GraphBuilder:
    """Build PPI + pathway graph for a gene set."""

    use_api: bool = True
    string_confidence: int = 700
    min_shared_pathways: int = 2
    cache_path: Optional[str] = None
    seed: int = 42

    def build(
        self,
        gene_ids: List[int],
        gene_names: List[str] | None = None,
    ) -> Data:
        if self.cache_path and Path(self.cache_path).exists():
            logger.info("Loading cached graph from %s", self.cache_path)
            return torch.load(self.cache_path, weights_only=False)

        # Build name-to-index mapping for API-based lookups
        name_to_idx: Dict[str, int] = {}
        if gene_names:
            name_to_idx = {name.upper(): i for i, name in enumerate(gene_names)}

        n_genes = len(gene_ids)

        if self.use_api and gene_names:
            ppi_edges = self._fetch_string_ppi(gene_names, name_to_idx)
            pathway_edges = self._build_pathway_edges(gene_names, name_to_idx)
        else:
            ppi_edges = self._synthetic_ppi(n_genes)
            pathway_edges = self._synthetic_pathway(n_genes)

        graph = self._assemble_graph(n_genes, ppi_edges, pathway_edges)

        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, self.cache_path)
            logger.info("Cached graph to %s", self.cache_path)

        return graph

    def _assemble_graph(
        self,
        n_genes: int,
        ppi_edges: List[Tuple[int, int]],
        pathway_edges: List[Tuple[int, int]],
    ) -> Data:
        all_edges = []
        all_types = []

        for si, di in ppi_edges:
            if 0 <= si < n_genes and 0 <= di < n_genes:
                all_edges.extend([(si, di), (di, si)])
                all_types.extend([EDGE_TYPE_PPI, EDGE_TYPE_PPI])

        for si, di in pathway_edges:
            if 0 <= si < n_genes and 0 <= di < n_genes:
                all_edges.extend([(si, di), (di, si)])
                all_types.extend([EDGE_TYPE_PATHWAY, EDGE_TYPE_PATHWAY])

        if not all_edges:
            logger.warning("No edges found, adding self-loops")
            all_edges = [(i, i) for i in range(n_genes)]
            all_types = [EDGE_TYPE_PPI] * n_genes

        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(all_types, dtype=torch.long)

        # Deduplicate
        seen: Set[Tuple[int, int, int]] = set()
        unique_mask = []
        for i in range(edge_index.shape[1]):
            key = (edge_index[0, i].item(), edge_index[1, i].item(), all_types[i])
            if key not in seen:
                seen.add(key)
                unique_mask.append(i)
        idx = torch.tensor(unique_mask, dtype=torch.long)
        edge_index = edge_index[:, idx]
        edge_type = edge_type[idx]

        return Data(
            num_nodes=n_genes,
            edge_index=edge_index,
            edge_type=edge_type,
        )

    def _fetch_string_ppi(
        self,
        gene_names: List[str],
        name_to_idx: Dict[str, int],
    ) -> List[Tuple[int, int]]:
        """Query STRING-DB using gene names and map results back to local indices."""
        from src.utils.http_utils import get_json_with_retry

        edges: List[Tuple[int, int]] = []
        # STRING API has a limit; batch in chunks of 200
        batch_size = 200
        for start in range(0, len(gene_names), batch_size):
            batch = gene_names[start:start + batch_size]
            try:
                identifiers = "%0d".join(batch)
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
                    # STRING returns preferredName_A/B which are gene symbols
                    name_a = row.get("preferredName_A", "").upper()
                    name_b = row.get("preferredName_B", "").upper()
                    idx_a = name_to_idx.get(name_a)
                    idx_b = name_to_idx.get(name_b)
                    if idx_a is not None and idx_b is not None and idx_a != idx_b:
                        edges.append((idx_a, idx_b))

                logger.info(
                    "STRING API batch %d-%d: %d interactions found",
                    start, start + len(batch), len(edges),
                )
            except Exception:
                logger.warning(
                    "STRING API failed for batch %d-%d, skipping",
                    start, start + len(batch),
                )

        if not edges:
            logger.warning("No STRING PPI edges found, using synthetic fallback")
            return self._synthetic_ppi(len(gene_names))

        return edges

    def _build_pathway_edges(
        self,
        gene_names: List[str],
        name_to_idx: Dict[str, int],
    ) -> List[Tuple[int, int]]:
        """Build pathway edges from hardcoded muscle/sarcopenia pathways."""
        edges: List[Tuple[int, int]] = []

        for pathway_name, pathway_genes in MUSCLE_PATHWAY_GENES.items():
            # Find which pathway genes are in our gene set
            member_indices = []
            for g in pathway_genes:
                idx = name_to_idx.get(g.upper())
                if idx is not None:
                    member_indices.append(idx)

            # Create edges between all members of this pathway
            for i, a in enumerate(member_indices):
                for b in member_indices[i + 1:]:
                    edges.append((a, b))

            if len(member_indices) >= 2:
                logger.info(
                    "Pathway '%s': %d/%d genes present, %d edges",
                    pathway_name, len(member_indices), len(pathway_genes),
                    len(member_indices) * (len(member_indices) - 1) // 2,
                )

        logger.info("Total pathway edges: %d", len(edges))
        return edges

    def _synthetic_ppi(self, n_genes: int) -> List[Tuple[int, int]]:
        """Synthetic PPI with denser connectivity (~3-5 neighbors per node)."""
        rng = torch.Generator().manual_seed(self.seed)
        edges = []
        for i in range(n_genes):
            # geometric with p=0.3 gives mean ~3.3 neighbors
            n_neighbors = int(torch.zeros(1).geometric_(0.3, generator=rng).item())
            n_neighbors = min(n_neighbors, 10)
            for _ in range(n_neighbors):
                j = int(torch.zeros(1).random_(0, n_genes, generator=rng).item())
                if i != j:
                    edges.append((i, j))
        return edges

    def _synthetic_pathway(self, n_genes: int) -> List[Tuple[int, int]]:
        """Synthetic pathway edges with denser clusters."""
        rng = torch.Generator().manual_seed(self.seed + 1)
        n_pathways = max(8, n_genes // 15)
        edges = []
        for _ in range(n_pathways):
            size = int(torch.zeros(1).random_(4, min(20, n_genes), generator=rng).item())
            members = torch.zeros(size, dtype=torch.long).random_(0, n_genes, generator=rng)
            member_list = members.unique().tolist()
            for i, a in enumerate(member_list):
                for b in member_list[i + 1:]:
                    edges.append((int(a), int(b)))
        return edges
