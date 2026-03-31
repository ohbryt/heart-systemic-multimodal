#!/usr/bin/env python3
"""
Fetch heart data from CELLxGENE without any API key.

Modes
-----
1) census-slice:
   Query CELLxGENE Census and save a filtered AnnData (.h5ad)
   Example: human + heart + primary data only + optional disease filter

2) source-h5ad:
   Download the original source H5AD for a known dataset_id

Requirements
------------
pip install cellxgene-census scanpy anndata pandas pyarrow scipy

Examples
--------
# 1) Save a heart slice from Census
python scripts/fetch_cellxgene_heart.py census-slice \
    --output data/raw/heart_human_primary.h5ad \
    --tissue-general heart \
    --primary-only

# 2) Save only failing heart-related cells if disease labels exist
python scripts/fetch_cellxgene_heart.py census-slice \
    --output data/raw/heart_failure_like.h5ad \
    --tissue-general heart \
    --disease "heart failure" \
    --primary-only

# 3) Download original source H5AD if you already know a dataset_id
python scripts/fetch_cellxgene_heart.py source-h5ad \
    --dataset-id e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb \
    --output data/raw/source_dataset.h5ad

# 4) Just print the cloud URI of the source h5ad
python scripts/fetch_cellxgene_heart.py source-uri \
    --dataset-id e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _import_census():
    try:
        import cellxgene_census
        return cellxgene_census
    except ImportError as e:
        raise SystemExit(
            "cellxgene-census is not installed.\n"
            "Install with:\n"
            "  pip install cellxgene-census scanpy anndata pandas pyarrow scipy"
        ) from e


def build_value_filter(
    tissue_general: str = "heart",
    disease: Optional[str] = None,
    primary_only: bool = True,
    assay: Optional[str] = None,
) -> str:
    """Build a Census obs filter string."""
    filters = [f"tissue_general == '{tissue_general}'"]
    if disease:
        filters.append(f"disease == '{disease}'")
    if primary_only:
        filters.append("is_primary_data == True")
    if assay:
        filters.append(f"assay == '{assay}'")
    return " and ".join(filters)


def save_obs_metadata_csv(adata, out_path: Path) -> None:
    """Save cell metadata to CSV alongside the h5ad."""
    meta_path = out_path.with_suffix(".obs.csv")
    adata.obs.to_csv(meta_path)
    logging.info("Saved obs metadata: %s", meta_path)


def census_slice_to_h5ad(
    output: Path,
    organism: str = "Homo sapiens",
    tissue_general: str = "heart",
    disease: Optional[str] = None,
    primary_only: bool = True,
    assay: Optional[str] = None,
    max_cells: Optional[int] = 50000,
    random_state: int = 42,
    measurement_name: str = "RNA",
) -> None:
    """
    Query a heart slice from CELLxGENE Census and save as h5ad.

    Notes
    -----
    - Census allows metadata-based slicing and returns AnnData objects.
    - For large selections, downsampling is recommended for local RAM.
    """
    cellxgene_census = _import_census()

    output.parent.mkdir(parents=True, exist_ok=True)

    value_filter = build_value_filter(
        tissue_general=tissue_general,
        disease=disease,
        primary_only=primary_only,
        assay=assay,
    )
    logging.info("Using value_filter: %s", value_filter)

    with cellxgene_census.open_soma() as census:
        logging.info("Connected to CELLxGENE Census (organism=%s)", organism)

        # Get cell count first
        obs_df = cellxgene_census.get_obs(
            census,
            organism,
            value_filter=value_filter,
            column_names=["soma_joinid", "cell_type", "disease", "tissue_general", "dataset_id"],
        )
        total_cells = len(obs_df)
        logging.info("Found %d cells matching filter", total_cells)

        if total_cells == 0:
            logging.warning("No cells found. Check filter: %s", value_filter)
            return

        # Downsample if needed
        obs_coords = None
        if max_cells and total_cells > max_cells:
            logging.info("Downsampling %d → %d cells", total_cells, max_cells)
            sampled = obs_df.sample(n=max_cells, random_state=random_state)
            obs_coords = sampled["soma_joinid"].tolist()
            del sampled

        # Fetch AnnData
        logging.info("Fetching AnnData from Census...")
        adata = cellxgene_census.get_anndata(
            census,
            organism=organism,
            measurement_name=measurement_name,
            obs_value_filter=value_filter,
            obs_coords=obs_coords,
        )

        logging.info("AnnData: %d cells × %d genes", adata.n_obs, adata.n_vars)

        # Save
        adata.write_h5ad(output)
        logging.info("Saved: %s (%.1f MB)", output, output.stat().st_size / 1e6)

        # Save metadata CSV
        save_obs_metadata_csv(adata, output)

        # Print summary
        if "cell_type" in adata.obs.columns:
            logging.info("Cell types:\n%s", adata.obs["cell_type"].value_counts().head(20).to_string())
        if "disease" in adata.obs.columns:
            logging.info("Diseases:\n%s", adata.obs["disease"].value_counts().to_string())


def download_source_h5ad(
    dataset_id: str,
    output: Path,
) -> None:
    """Download the original source H5AD for a known dataset_id."""
    cellxgene_census = _import_census()

    output.parent.mkdir(parents=True, exist_ok=True)

    with cellxgene_census.open_soma() as census:
        logging.info("Looking up source h5ad for dataset_id=%s", dataset_id)

        try:
            uri = cellxgene_census.download_source_h5ad(
                dataset_id,
                to_path=str(output),
            )
            logging.info("Downloaded source h5ad: %s", output)
        except Exception as e:
            logging.error("Failed to download source h5ad: %s", e)

            # Fallback: try Discover API
            logging.info("Trying CELLxGENE Discover API fallback...")
            import requests

            api_url = f"https://api.cellxgene.cziscience.com/curation/v1/datasets/{dataset_id}"
            resp = requests.get(api_url, timeout=30)
            if resp.ok:
                data = resp.json()
                assets = data.get("assets", [])
                h5ad_assets = [a for a in assets if a.get("filetype") == "H5AD"]
                if h5ad_assets:
                    download_url = h5ad_assets[0]["url"]
                    logging.info("Downloading from: %s", download_url)

                    with requests.get(download_url, stream=True, timeout=600) as r:
                        r.raise_for_status()
                        with open(output, "wb") as f:
                            for chunk in r.iter_content(chunk_size=64 * 1024 * 1024):
                                f.write(chunk)

                    logging.info("Downloaded: %s (%.1f MB)", output, output.stat().st_size / 1e6)
                else:
                    logging.error("No H5AD assets found for dataset_id=%s", dataset_id)
            else:
                logging.error("Discover API failed: %s", resp.text)


def print_source_uri(dataset_id: str) -> None:
    """Print the cloud URI of the source h5ad."""
    cellxgene_census = _import_census()

    with cellxgene_census.open_soma() as census:
        try:
            uri = cellxgene_census.get_source_h5ad_uri(dataset_id)
            print(f"Source H5AD URI: {uri}")
        except Exception as e:
            logging.error("Failed to get source URI: %s", e)

            # Fallback
            import requests
            api_url = f"https://api.cellxgene.cziscience.com/curation/v1/datasets/{dataset_id}"
            resp = requests.get(api_url, timeout=30)
            if resp.ok:
                data = resp.json()
                assets = data.get("assets", [])
                h5ad_assets = [a for a in assets if a.get("filetype") == "H5AD"]
                if h5ad_assets:
                    print(f"Download URL: {h5ad_assets[0]['url']}")
                else:
                    print("No H5AD assets found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch heart data from CELLxGENE (no API key needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # census-slice
    p_slice = subparsers.add_parser("census-slice", help="Query Census and save filtered h5ad")
    p_slice.add_argument("--output", type=Path, required=True)
    p_slice.add_argument("--organism", default="Homo sapiens")
    p_slice.add_argument("--tissue-general", default="heart")
    p_slice.add_argument("--disease", default=None)
    p_slice.add_argument("--assay", default=None)
    p_slice.add_argument("--primary-only", action="store_true", default=True)
    p_slice.add_argument("--max-cells", type=int, default=50000)
    p_slice.add_argument("--seed", type=int, default=42)

    # source-h5ad
    p_source = subparsers.add_parser("source-h5ad", help="Download original source H5AD")
    p_source.add_argument("--dataset-id", required=True)
    p_source.add_argument("--output", type=Path, required=True)

    # source-uri
    p_uri = subparsers.add_parser("source-uri", help="Print cloud URI of source h5ad")
    p_uri.add_argument("--dataset-id", required=True)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "census-slice":
        census_slice_to_h5ad(
            output=args.output,
            organism=args.organism,
            tissue_general=args.tissue_general,
            disease=args.disease,
            primary_only=args.primary_only,
            assay=args.assay,
            max_cells=args.max_cells,
            random_state=args.seed,
        )
    elif args.command == "source-h5ad":
        download_source_h5ad(args.dataset_id, args.output)
    elif args.command == "source-uri":
        print_source_uri(args.dataset_id)


if __name__ == "__main__":
    main()
