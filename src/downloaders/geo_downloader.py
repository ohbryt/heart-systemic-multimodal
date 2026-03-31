"""
GEO (Gene Expression Omnibus) dataset downloader.

Downloads supplementary files for:
- GSE183852: Heart single-cell RNA-seq
- GSE135805: Cardiac single-cell atlas
- GSE290577: Additional heart SC dataset
"""

import ftplib
import hashlib
import logging
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urlparse

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

# GEO accessions to download
GEO_ACCESSIONS: Dict[str, Dict] = {
    "GSE183852": {
        "description": "Human heart single-cell RNA-seq (systemic/EV study)",
        "expected_files": ["*.h5ad", "*.mtx.gz", "*.tsv.gz", "barcodes*", "features*", "matrix*"],
    },
    "GSE135805": {
        "description": "Cardiac cell atlas single-cell RNA-seq",
        "expected_files": ["*.h5ad", "*.h5", "*.mtx.gz"],
    },
    "GSE290577": {
        "description": "Heart single-cell supplementary dataset",
        "expected_files": ["*.h5ad", "*.mtx.gz", "*.tsv.gz"],
    },
}

GEO_FTP_BASE = "ftp.ncbi.nlm.nih.gov"
GEO_HTTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"


class DownloadProgress:
    """Tracks download progress with optional tqdm."""

    def __init__(self, filename: str, total: Optional[int] = None) -> None:
        self.filename = filename
        self.total = total
        self._downloaded = 0
        if HAS_TQDM:
            self._bar = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                leave=True,
            )
        else:
            self._bar = None

    def update(self, chunk: int) -> None:
        self._downloaded += chunk
        if self._bar is not None:
            self._bar.update(chunk)
        else:
            if self.total:
                pct = self._downloaded / self.total * 100
                print(f"\r  {self.filename}: {pct:.1f}%", end="", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        else:
            print()


class GEODownloader:
    """
    Downloads GEO datasets with FTP/HTTP fallback and local caching.

    Usage
    -----
    >>> dl = GEODownloader(cache_dir=Path("data/raw/geo"))
    >>> paths = dl.download_all()
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/raw/geo"),
        accessions: Optional[List[str]] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        cache_dir:
            Directory to store downloaded files.
        accessions:
            GEO accession IDs to download. Defaults to all defined accessions.
        timeout:
            Connection timeout in seconds.
        max_retries:
            Number of retry attempts on failure.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.accessions = accessions or list(GEO_ACCESSIONS.keys())
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_all(self) -> Dict[str, Path]:
        """
        Download all configured GEO accessions.

        Returns
        -------
        Dict mapping accession ID -> local directory path.
        """
        results: Dict[str, Path] = {}
        for acc in self.accessions:
            logger.info("Processing accession: %s", acc)
            try:
                local_dir = self.download_accession(acc)
                results[acc] = local_dir
            except Exception as exc:
                logger.error("Failed to download %s: %s", acc, exc)
                self._print_manual_instructions(acc)
        return results

    def download_accession(self, accession: str) -> Path:
        """
        Download supplementary files for a single GEO accession.

        Parameters
        ----------
        accession:
            GEO series accession (e.g. ``GSE183852``).

        Returns
        -------
        Local directory containing downloaded files.
        """
        acc_dir = self.cache_dir / accession
        acc_dir.mkdir(parents=True, exist_ok=True)

        # Check cache first
        cached = list(acc_dir.iterdir())
        if cached:
            logger.info("Found %d cached files for %s", len(cached), accession)
            return acc_dir

        file_list = self._get_file_list_ftp(accession)
        if not file_list:
            logger.warning("No files found via FTP for %s, trying HTTP listing", accession)
            file_list = self._get_file_list_http(accession)

        if not file_list:
            logger.error("Could not retrieve file list for %s", accession)
            self._print_manual_instructions(accession)
            return acc_dir

        for remote_url, filename in file_list:
            dest = acc_dir / filename
            if dest.exists():
                logger.info("  Skipping (cached): %s", filename)
                continue
            self._download_file(remote_url, dest)

        return acc_dir

    # ------------------------------------------------------------------
    # FTP helpers
    # ------------------------------------------------------------------

    def _get_file_list_ftp(self, accession: str) -> List[Tuple[str, str]]:
        """
        Retrieve supplementary file list via FTP.

        Returns list of (url, filename) tuples.
        """
        # GEO FTP path: /geo/series/GSEnnn/GSExxxxx/suppl/
        prefix = accession[:-3] + "nnn"  # e.g. GSE183nnn
        ftp_path = f"/geo/series/{prefix}/{accession}/suppl/"

        files: List[Tuple[str, str]] = []
        try:
            with ftplib.FTP(GEO_FTP_BASE, timeout=self.timeout) as ftp:
                ftp.login()
                try:
                    ftp.cwd(ftp_path)
                    names = ftp.nlst()
                except ftplib.error_perm as exc:
                    logger.debug("FTP path not found %s: %s", ftp_path, exc)
                    return []

                for name in names:
                    url = f"https://{GEO_FTP_BASE}{ftp_path}{name}"
                    files.append((url, name))

        except Exception as exc:
            logger.debug("FTP connection failed: %s", exc)

        return files

    def _get_file_list_http(self, accession: str) -> List[Tuple[str, str]]:
        """
        Retrieve supplementary file list via HTTP (fallback).

        Returns list of (url, filename) tuples.
        """
        prefix = accession[:-3] + "nnn"
        url = f"{GEO_HTTP_BASE}/{prefix}/{accession}/suppl/"

        files: List[Tuple[str, str]] = []
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Simple href extraction (GEO uses plain Apache directory listings)
            import re
            for match in re.finditer(r'href="([^"?/][^"]*)"', html):
                name = match.group(1)
                if name.startswith("GSE"):
                    file_url = url + name
                    files.append((file_url, name))

        except Exception as exc:
            logger.debug("HTTP listing failed for %s: %s", url, exc)

        return files

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_file(self, url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
        """
        Download a single file with retries and progress tracking.

        Parameters
        ----------
        url:
            Remote URL.
        dest:
            Local destination path.
        chunk_size:
            Read chunk size in bytes (default 1 MB).
        """
        filename = dest.name
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "GEODownloader/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    total = int(resp.headers.get("Content-Length", 0)) or None
                    progress = DownloadProgress(filename, total)
                    try:
                        with open(tmp_path, "wb") as fh:
                            while True:
                                chunk = resp.read(chunk_size)
                                if not chunk:
                                    break
                                fh.write(chunk)
                                progress.update(len(chunk))
                    finally:
                        progress.close()

                tmp_path.rename(dest)
                logger.info("  Downloaded: %s", filename)
                return

            except (URLError, OSError) as exc:
                logger.warning("  Attempt %d/%d failed for %s: %s", attempt, self.max_retries, filename, exc)
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("  All attempts failed for %s", filename)

    # ------------------------------------------------------------------
    # Fallback instructions
    # ------------------------------------------------------------------

    def _print_manual_instructions(self, accession: str) -> None:
        """Print manual download instructions for an accession."""
        info = GEO_ACCESSIONS.get(accession, {})
        desc = info.get("description", "")
        dest = self.cache_dir / accession

        print(
            f"\n{'='*60}\n"
            f"MANUAL DOWNLOAD REQUIRED: {accession}\n"
            f"Description: {desc}\n"
            f"{'='*60}\n"
            f"1. Go to: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}\n"
            f"2. Scroll to 'Supplementary file' section\n"
            f"3. Download all .h5ad, .mtx.gz, or .tsv.gz files\n"
            f"4. Place downloaded files in: {dest.absolute()}\n"
            f"{'='*60}\n"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def verify_checksums(self, accession: str) -> Dict[str, bool]:
        """
        Verify MD5 checksums if a checksum file is present.

        Returns dict of filename -> is_valid.
        """
        acc_dir = self.cache_dir / accession
        results: Dict[str, bool] = {}

        checksum_file = acc_dir / "checksums.md5"
        if not checksum_file.exists():
            logger.info("No checksum file found for %s", accession)
            return results

        for line in checksum_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            expected_md5, fname = parts
            fpath = acc_dir / fname
            if not fpath.exists():
                results[fname] = False
                continue

            actual_md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
            results[fname] = actual_md5 == expected_md5

        return results

    def list_downloaded(self) -> Dict[str, List[Path]]:
        """Return dict of accession -> list of local file paths."""
        result: Dict[str, List[Path]] = {}
        for acc in self.accessions:
            acc_dir = self.cache_dir / acc
            if acc_dir.exists():
                result[acc] = sorted(acc_dir.iterdir())
            else:
                result[acc] = []
        return result
