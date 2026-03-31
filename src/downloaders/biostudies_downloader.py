"""
BioStudies downloader.

Downloads datasets from EMBL-EBI BioStudies:
- E-MTAB-15659: Optional brain reference single-cell dataset

Uses the BioStudies REST API to enumerate files, then downloads
relevant supplementary data.
"""

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

BIOSTUDIES_API_BASE = "https://www.ebi.ac.uk/biostudies/api/v1"
BIOSTUDIES_FILES_BASE = "https://ftp.ebi.ac.uk/biostudies/fire"

# Accessions to download
BIOSTUDIES_ACCESSIONS: Dict[str, Dict] = {
    "E-MTAB-15659": {
        "description": "Brain reference single-cell RNA-seq (optional reference atlas)",
        "preferred_extensions": [".h5ad", ".h5", ".loom", ".mtx.gz", ".tsv.gz"],
        "optional": True,
    },
}

# Large raw file extensions to skip by default
SKIP_EXTENSIONS = {".fastq", ".fastq.gz", ".bam", ".bai", ".cram", ".vcf", ".bcf"}


class BioStudiesDownloader:
    """
    Downloads files from EMBL-EBI BioStudies via the public REST API.

    Usage
    -----
    >>> dl = BioStudiesDownloader(cache_dir=Path("data/raw/biostudies"))
    >>> paths = dl.download_all()
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/raw/biostudies"),
        accessions: Optional[List[str]] = None,
        timeout: int = 60,
        max_retries: int = 3,
        skip_raw: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        cache_dir:
            Local directory for downloaded files.
        accessions:
            BioStudies accession IDs to download.
        timeout:
            HTTP connection timeout in seconds.
        max_retries:
            Number of retry attempts on failure.
        skip_raw:
            If True, skip raw sequencing files (.fastq, .bam, etc.).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.accessions = accessions or list(BIOSTUDIES_ACCESSIONS.keys())
        self.timeout = timeout
        self.max_retries = max_retries
        self.skip_raw = skip_raw

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_all(self) -> Dict[str, List[Path]]:
        """
        Download all configured BioStudies accessions.

        Returns
        -------
        Dict mapping accession -> list of downloaded file paths.
        """
        results: Dict[str, List[Path]] = {}
        for acc in self.accessions:
            info = BIOSTUDIES_ACCESSIONS.get(acc, {})
            is_optional = info.get("optional", False)

            logger.info(
                "Processing BioStudies accession: %s%s",
                acc,
                " (optional)" if is_optional else "",
            )
            try:
                paths = self.download_accession(acc)
                results[acc] = paths
            except Exception as exc:
                if is_optional:
                    logger.warning("Optional accession %s failed (skipping): %s", acc, exc)
                else:
                    logger.error("Failed to download %s: %s", acc, exc)
                self._print_manual_instructions(acc)
                results[acc] = []

        return results

    def download_accession(self, accession: str) -> List[Path]:
        """
        Download all relevant files for a single BioStudies accession.

        Parameters
        ----------
        accession:
            BioStudies accession ID (e.g. ``E-MTAB-15659``).

        Returns
        -------
        List of downloaded local file paths.
        """
        acc_dir = self.cache_dir / accession
        acc_dir.mkdir(parents=True, exist_ok=True)

        # Check cache
        cached = [p for p in acc_dir.rglob("*") if p.is_file()]
        if cached:
            logger.info("Using %d cached files for %s", len(cached), accession)
            return cached

        # Get file list from API
        file_list = self._get_file_list(accession)
        if not file_list:
            logger.error("No files found for %s", accession)
            self._print_manual_instructions(accession)
            return []

        prefs = BIOSTUDIES_ACCESSIONS.get(accession, {}).get("preferred_extensions", [])
        downloaded: List[Path] = []

        for url, rel_path in file_list:
            filename = Path(rel_path).name
            ext = "".join(Path(filename).suffixes).lower()
            plain_ext = Path(filename).suffix.lower()

            if self.skip_raw and (plain_ext in SKIP_EXTENSIONS or ext.startswith(".fastq")):
                logger.debug("Skipping raw file: %s", filename)
                continue

            # Prefer specific extensions if defined
            if prefs:
                matched = any(filename.endswith(p) for p in prefs)
                if not matched:
                    logger.debug("Skipping non-preferred file: %s", filename)
                    continue

            dest = acc_dir / filename
            if dest.exists():
                logger.info("  Cached: %s", filename)
                downloaded.append(dest)
                continue

            path = self._download_file(url, dest)
            if path:
                downloaded.append(path)

        return downloaded

    def get_study_metadata(self, accession: str) -> Dict:
        """
        Retrieve study-level metadata from the BioStudies API.

        Parameters
        ----------
        accession:
            BioStudies accession ID.

        Returns
        -------
        Dict with study metadata (title, authors, abstract, etc.).
        """
        url = f"{BIOSTUDIES_API_BASE}/studies/{accession}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BioStudiesDownloader/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
            return data
        except Exception as exc:
            logger.warning("Could not fetch metadata for %s: %s", accession, exc)
            return {}

    # ------------------------------------------------------------------
    # BioStudies REST API
    # ------------------------------------------------------------------

    def _get_file_list(self, accession: str) -> List[Tuple[str, str]]:
        """
        Retrieve the list of files via BioStudies REST API.

        Returns list of (download_url, relative_path) tuples.
        """
        url = f"{BIOSTUDIES_API_BASE}/studies/{accession}/files/list"
        files: List[Tuple[str, str]] = []

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BioStudiesDownloader/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())

            for entry in data:
                rel_path = entry.get("path", entry.get("name", ""))
                if not rel_path:
                    continue

                # Construct download URL
                download_url = self._build_download_url(accession, rel_path)
                files.append((download_url, rel_path))

        except Exception as exc:
            logger.debug("File list API failed for %s: %s", accession, exc)
            # Try the FTP listing approach
            files = self._get_file_list_ftp(accession)

        return files

    def _get_file_list_ftp(self, accession: str) -> List[Tuple[str, str]]:
        """
        Fallback: retrieve file list via BioStudies FTP HTTP index.

        Returns list of (url, filename) tuples.
        """
        import re

        # BioStudies FTP structure for ArrayExpress-style accessions
        # e.g. E-MTAB-15659 -> /biostudies/fire/E-MTAB-nnn/E-MTAB-15659/
        prefix = accession[:7] + "nnn"  # E-MTAB-nnn
        ftp_url = f"{BIOSTUDIES_FILES_BASE}/{prefix}/{accession}/"

        files: List[Tuple[str, str]] = []
        try:
            req = urllib.request.Request(
                ftp_url,
                headers={"User-Agent": "BioStudiesDownloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            for m in re.finditer(r'href="([^"/][^"?#]*)"', html):
                name = m.group(1)
                files.append((ftp_url + name, name))

        except Exception as exc:
            logger.debug("FTP index failed for %s: %s", accession, exc)

        return files

    def _build_download_url(self, accession: str, rel_path: str) -> str:
        """
        Build the FTP/HTTPS download URL for a given file path.

        Parameters
        ----------
        accession:
            BioStudies accession.
        rel_path:
            Relative path from the API response.

        Returns
        -------
        Full download URL string.
        """
        # BioStudies uses fire.ebi.ac.uk for file downloads
        prefix = accession[:7] + "nnn"
        clean_path = rel_path.lstrip("/")
        return f"{BIOSTUDIES_FILES_BASE}/{prefix}/{accession}/{clean_path}"

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_file(
        self, url: str, dest: Path, chunk_size: int = 512 * 1024
    ) -> Optional[Path]:
        """
        Download a single file with progress reporting and retries.

        Parameters
        ----------
        url:
            Remote URL.
        dest:
            Local destination path.
        chunk_size:
            Read chunk size in bytes.

        Returns
        -------
        dest on success, None on failure.
        """
        filename = dest.name
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "BioStudiesDownloader/1.0"},
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
                                chunk = resp.read(chunk_size)
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
                logger.info("  Downloaded: %s (%.1f MB)", filename, dest.stat().st_size / 1e6)
                return dest

            except (URLError, OSError) as exc:
                logger.warning(
                    "  Attempt %d/%d failed for %s: %s",
                    attempt, self.max_retries, filename, exc,
                )
                if tmp.exists():
                    tmp.unlink()
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("  All attempts failed for %s", filename)
        return None

    # ------------------------------------------------------------------
    # Manual instructions
    # ------------------------------------------------------------------

    def _print_manual_instructions(self, accession: str) -> None:
        """Print manual download instructions for a BioStudies accession."""
        info = BIOSTUDIES_ACCESSIONS.get(accession, {})
        dest = self.cache_dir / accession
        print(
            f"\n{'='*60}\n"
            f"MANUAL DOWNLOAD REQUIRED: {accession}\n"
            f"Description: {info.get('description', '')}\n"
            f"Optional: {info.get('optional', False)}\n"
            f"{'='*60}\n"
            f"1. Go to: https://www.ebi.ac.uk/biostudies/studies/{accession}\n"
            f"2. Click 'Files' tab\n"
            f"3. Download files with extensions: {info.get('preferred_extensions', [])}\n"
            f"4. Place files in: {dest.absolute()}\n"
            f"{'='*60}\n"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_downloaded(self) -> Dict[str, List[Path]]:
        """Return dict of accession -> list of downloaded file paths."""
        result: Dict[str, List[Path]] = {}
        for acc in self.accessions:
            acc_dir = self.cache_dir / acc
            if acc_dir.exists():
                result[acc] = sorted(p for p in acc_dir.rglob("*") if p.is_file())
            else:
                result[acc] = []
        return result
