from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data

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
        id_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
        all_edges = []
        all_types = []

        for src, dst in ppi_edges:
            if src in id_to_idx and dst in id_to_idx:
                si, di = id_to_idx[src], id_to_idx[dst]
                all_edges.extend([(si, di), (di, si)])
                all_types.extend([EDGE_TYPE_PPI, EDGE_TYPE_PPI])

        for src, dst in pathway_edges:
            if src in id_to_idx and dst in id_to_idx:
                si, di = id_to_idx[src], id_to_idx[dst]
                all_edges.extend([(si, di), (di, si)])
                all_types.extend([EDGE_TYPE_PATHWAY, EDGE_TYPE_PATHWAY])

        if not all_edges:
            all_edges = [(i, i) for i in range(len(gene_ids))]
            all_types = [EDGE_TYPE_PPI] * len(gene_ids)

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
            num_nodes=len(gene_ids),
            edge_index=edge_index,
            edge_type=edge_type,
        )

    def _fetch_string_ppi(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        from src.utils.http_utils import get_json_with_retry
        edges = []
        try:
            identifiers = "%0d".join(str(g) for g in gene_ids[:200])
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
                if "stringId_A" in row and "stringId_B" in row:
                    edges.append((hash(row["stringId_A"]) % 100000,
                                  hash(row["stringId_B"]) % 100000))
        except Exception:
            logger.warning("STRING API failed, returning empty PPI edges")
        return edges

    def _fetch_kegg_pathway_edges(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        logger.info("KEGG pathway fetch not yet implemented, returning empty")
        return []

    def _synthetic_ppi(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        rng = torch.Generator().manual_seed(self.seed)
        n = len(gene_ids)
        edges = []
        for i in range(n):
            n_neighbors = int(torch.zeros(1).geometric_(0.2, generator=rng).item())
            n_neighbors = min(n_neighbors, 8)
            for _ in range(n_neighbors):
                j = int(torch.zeros(1).random_(0, n, generator=rng).item())
                if i != j:
                    edges.append((gene_ids[i], gene_ids[j]))
        return edges

    def _synthetic_pathway(self, gene_ids: List[int]) -> List[Tuple[int, int]]:
        rng = torch.Generator().manual_seed(self.seed + 1)
        n = len(gene_ids)
        n_pathways = max(5, n // 20)
        edges = []
        for _ in range(n_pathways):
            size = int(torch.zeros(1).random_(3, min(15, n), generator=rng).item())
            members = torch.zeros(size, dtype=torch.long).random_(0, n, generator=rng)
            member_list = members.unique().tolist()
            for i, a in enumerate(member_list):
                for b in member_list[i + 1:]:
                    edges.append((gene_ids[a], gene_ids[b]))
        return edges
