"""
CELLxGENE Census downloader.

Downloads single-cell datasets from the Chan Zuckerberg CELLxGENE portal.

Collection: e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb
(Human heart / cardiovascular single-cell data)
"""

import logging
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

CELLXGENE_API_BASE = "https://api.cellxgene.cziscience.com"
CELLXGENE_COLLECTION_ID = "e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb"


class CellxGeneDownloader:
    """
    Downloads datasets from CELLxGENE via the census API or direct HTTP.

    Supports memory-aware streaming for large h5ad files.

    Usage
    -----
    >>> dl = CellxGeneDownloader(cache_dir=Path("data/raw/cellxgene"))
    >>> paths = dl.download_collection()
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/raw/cellxgene"),
        collection_id: str = CELLXGENE_COLLECTION_ID,
        timeout: int = 120,
        max_retries: int = 3,
        stream_chunk_mb: int = 64,
    ) -> None:
        """
        Parameters
        ----------
        cache_dir:
            Local directory for cached files.
        collection_id:
            CELLxGENE collection UUID.
        timeout:
            HTTP timeout in seconds.
        max_retries:
            Retry count on transient failures.
        stream_chunk_mb:
            Chunk size (MB) for streaming downloads.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.collection_id = collection_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.stream_chunk = stream_chunk_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_collection(
        self,
        file_format: str = "H5AD",
        organism: str = "Homo sapiens",
        tissue_filter: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Download all datasets in the collection matching the given filters.

        Parameters
        ----------
        file_format:
            Preferred file format (``H5AD`` or ``RDS``).
        organism:
            Filter by organism name.
        tissue_filter:
            If set, only download datasets whose tissue matches any entry.

        Returns
        -------
        List of downloaded local file paths.
        """
        # Try census API first, fall back to portal API
        datasets = self._get_datasets_census(organism, tissue_filter)
        if not datasets:
            logger.info("Census API unavailable or no results; trying portal REST API")
            datasets = self._get_datasets_rest(file_format, organism, tissue_filter)

        if not datasets:
            logger.error("No datasets found for collection %s", self.collection_id)
            self._print_manual_instructions()
            return []

        logger.info("Found %d dataset(s) to download", len(datasets))
        downloaded: List[Path] = []

        for ds in datasets:
            try:
                path = self._download_dataset(ds, file_format)
                if path:
                    downloaded.append(path)
            except Exception as exc:
                logger.error("Failed to download dataset %s: %s", ds.get("dataset_id", "?"), exc)

        return downloaded

    def download_by_dataset_id(self, dataset_id: str, file_format: str = "H5AD") -> Optional[Path]:
        """
        Download a specific dataset by its CELLxGENE dataset ID.

        Parameters
        ----------
        dataset_id:
            CELLxGENE dataset UUID.
        file_format:
            ``H5AD`` or ``RDS``.

        Returns
        -------
        Local path to the downloaded file, or None on failure.
        """
        url = self._get_download_url_rest(dataset_id, file_format)
        if not url:
            logger.error("Could not resolve download URL for dataset %s", dataset_id)
            return None

        filename = f"{dataset_id}.{file_format.lower()}"
        dest = self.cache_dir / filename
        if dest.exists():
            logger.info("Cached: %s", dest)
            return dest

        return self._stream_download(url, dest)

    # ------------------------------------------------------------------
    # Census API (cellxgene-census package)
    # ------------------------------------------------------------------

    def _get_datasets_census(
        self,
        organism: str,
        tissue_filter: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Use cellxgene-census Python package to query dataset metadata.

        Returns empty list if package is not installed.
        """
        try:
            import cellxgene_census  # type: ignore
        except ImportError:
            logger.debug("cellxgene-census package not available")
            return []

        datasets: List[Dict[str, Any]] = []
        try:
            with cellxgene_census.open_soma() as census:
                dataset_df = cellxgene_census.get_dataset_metadata(census)
                # Filter to our collection
                mask = dataset_df["collection_id"] == self.collection_id
                if organism:
                    mask &= dataset_df["organism_ontology_term_id"].str.contains(
                        "NCBITaxon:9606" if "sapiens" in organism else organism,
                        case=False,
                        na=False,
                    )
                filtered = dataset_df[mask]

                for _, row in filtered.iterrows():
                    entry: Dict[str, Any] = row.to_dict()
                    if tissue_filter:
                        tissue_val = str(entry.get("tissue", "")).lower()
                        if not any(t.lower() in tissue_val for t in tissue_filter):
                            continue
                    datasets.append(entry)

        except Exception as exc:
            logger.warning("Census API query failed: %s", exc)

        return datasets

    # ------------------------------------------------------------------
    # Portal REST API
    # ------------------------------------------------------------------

    def _get_datasets_rest(
        self,
        file_format: str,
        organism: str,
        tissue_filter: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Query CELLxGENE collection via the public REST API.

        Returns list of dataset metadata dicts.
        """
        import json

        url = f"{CELLXGENE_API_BASE}/dp/v1/collections/{self.collection_id}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "CellxGeneDownloader/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.error("REST API request failed: %s", exc)
            return []

        datasets: List[Dict[str, Any]] = []
        for ds in data.get("datasets", []):
            # Apply organism filter
            org_name = ds.get("organism", [{}])
            if isinstance(org_name, list):
                org_label = " ".join(o.get("label", "") for o in org_name)
            else:
                org_label = str(org_name)

            if organism and organism.lower() not in org_label.lower():
                continue

            # Apply tissue filter
            if tissue_filter:
                tissue_labels = [
                    t.get("label", "").lower()
                    for t in ds.get("tissue", [])
                    if isinstance(t, dict)
                ]
                if not any(
                    any(f.lower() in tl for tl in tissue_labels)
                    for f in tissue_filter
                ):
                    continue

            ds["_preferred_format"] = file_format
            datasets.append(ds)

        return datasets

    def _get_download_url_rest(self, dataset_id: str, file_format: str) -> Optional[str]:
        """
        Resolve download URL for a dataset via REST API.

        Returns download URL or None.
        """
        import json

        url = f"{CELLXGENE_API_BASE}/dp/v1/datasets/{dataset_id}/assets"
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "CellxGeneDownloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.error("Could not fetch assets for %s: %s", dataset_id, exc)
            return None

        for asset in data.get("assets", []):
            if asset.get("filetype", "").upper() == file_format.upper():
                return asset.get("presigned_url") or asset.get("url")

        return None

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_dataset(self, dataset: Dict[str, Any], file_format: str) -> Optional[Path]:
        """
        Resolve and download a single dataset entry.

        Returns local file path or None.
        """
        dataset_id = dataset.get("dataset_id") or dataset.get("id", "")
        if not dataset_id:
            logger.warning("Dataset entry missing ID: %s", dataset)
            return None

        # Try census-style download URL first
        download_url: Optional[str] = None
        for asset in dataset.get("assets", []):
            if isinstance(asset, dict) and asset.get("filetype", "").upper() == file_format.upper():
                download_url = asset.get("presigned_url") or asset.get("url")
                break

        if not download_url:
            download_url = self._get_download_url_rest(dataset_id, file_format)

        if not download_url:
            logger.error("No download URL found for dataset %s", dataset_id)
            return None

        title = dataset.get("title", dataset_id)[:40].replace("/", "_")
        filename = f"{dataset_id}_{title}.{file_format.lower()}"
        dest = self.cache_dir / filename

        if dest.exists():
            logger.info("Cached: %s", dest)
            return dest

        logger.info("Downloading dataset '%s' -> %s", title, filename)
        return self._stream_download(download_url, dest)

    def _stream_download(self, url: str, dest: Path) -> Optional[Path]:
        """
        Stream-download a (potentially large) file in chunks.

        Writes to a .tmp file first, then atomically renames on success.

        Parameters
        ----------
        url:
            Remote URL.
        dest:
            Final local destination.

        Returns
        -------
        dest path on success, None on failure.
        """
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        filename = dest.name

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "CellxGeneDownloader/1.0"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    total = int(resp.headers.get("Content-Length", 0)) or None

                    if HAS_TQDM:
                        bar = tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=filename[:50],
                        )
                    else:
                        bar = None

                    try:
                        with open(tmp, "wb") as fh:
                            downloaded = 0
                            while True:
                                chunk = resp.read(self.stream_chunk)
                                if not chunk:
                                    break
                                fh.write(chunk)
                                downloaded += len(chunk)
                                if bar:
                                    bar.update(len(chunk))
                                elif total:
                                    pct = downloaded / total * 100
                                    print(f"\r  {filename[:40]}: {pct:.1f}%", end="", flush=True)
                    finally:
                        if bar:
                            bar.close()
                        else:
                            print()

                tmp.rename(dest)
                logger.info("Saved: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
                return dest

            except (URLError, OSError) as exc:
                logger.warning("Attempt %d/%d failed for %s: %s", attempt, self.max_retries, filename, exc)
                if tmp.exists():
                    tmp.unlink()
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("All download attempts failed for %s", filename)
        return None

    # ------------------------------------------------------------------
    # Fallback instructions
    # ------------------------------------------------------------------

    def _print_manual_instructions(self) -> None:
        """Print manual download instructions."""
        print(
            f"\n{'='*60}\n"
            f"MANUAL DOWNLOAD REQUIRED: CELLxGENE Collection\n"
            f"{'='*60}\n"
            f"Collection ID: {self.collection_id}\n"
            f"URL: https://cellxgene.cziscience.com/collections/{self.collection_id}\n"
            f"\nSteps:\n"
            f"1. Open the URL above in a browser\n"
            f"2. Click 'Download' on each dataset of interest\n"
            f"3. Select H5AD format\n"
            f"4. Move downloaded files to: {self.cache_dir.absolute()}\n"
            f"\nAlternatively, install the census package:\n"
            f"  pip install cellxgene-census\n"
            f"  python -c \"import cellxgene_census; cellxgene_census.download_source_h5ad('<dataset_id>', to_path='.')\"\n"
            f"{'='*60}\n"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_downloaded(self) -> List[Path]:
        """Return list of all downloaded files in the cache directory."""
        return sorted(p for p in self.cache_dir.iterdir() if p.is_file() and not p.suffix.endswith(".tmp"))
