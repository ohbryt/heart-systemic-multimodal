"""CELLxGENE pipeline stage — fetches heart data from Census."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_cellxgene_stage(
    datasets: list[dict[str, Any]],
    raw_dir: str | Path = "data/raw",
    manifest_dir: str | Path = "data/manifests",
) -> list[dict[str, Any]]:
    """
    Run CELLxGENE download stage for all enabled cellxgene datasets.

    Parameters
    ----------
    datasets : list[dict]
        Full dataset list from datasets.yaml
    raw_dir : str or Path
        Base directory for raw data
    manifest_dir : str or Path
        Directory for manifest files

    Returns
    -------
    list of result dicts with keys: id, status, output_path, cells, error
    """
    from src.downloaders.fetch_cellxgene import fetch_from_config

    raw_dir = Path(raw_dir)
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    cellxgene_datasets = [
        ds for ds in datasets
        if ds.get("type") == "cellxgene" and ds.get("enabled", True)
    ]

    if not cellxgene_datasets:
        logger.info("No enabled CELLxGENE datasets found. Skipping stage.")
        return []

    logger.info("CELLxGENE stage: %d datasets to fetch", len(cellxgene_datasets))
    results = []

    for ds in cellxgene_datasets:
        logger.info("Fetching CELLxGENE dataset: %s (mode=%s)", ds["id"], ds.get("mode", "?"))
        result = fetch_from_config(ds, base_dir=raw_dir)
        results.append(result)
        logger.info(
            "  → %s: %s (%d cells)",
            result["id"], result["status"], result.get("cells", 0),
        )

    # Save manifest
    manifest_path = manifest_dir / "cellxgene_manifest.json"
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_datasets": len(cellxgene_datasets),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("CELLxGENE manifest saved: %s", manifest_path)

    return results
