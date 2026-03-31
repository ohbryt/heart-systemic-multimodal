"""
PRIDE proteomics dataset downloader.

Downloads processed results from PRIDE Archive for:
- PXD021371: Human heart EV proteomics
- PXD059929: Cardiac proteomics dataset
- PXD060680: Systemic EV proteomics

Uses the PRIDE REST API to discover files, then downloads processed/result
files (avoiding large raw .raw/.mzML instrument files).
"""

import io
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError

import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = logging.getLogger(__name__)

PRIDE_API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"
PRIDE_FTP_BASE = "https://ftp.pride.ebi.ac.uk/pride/data/archive"

# Projects to download
PRIDE_PROJECTS: Dict[str, Dict] = {
    "PXD021371": {
        "description": "Human heart EV proteomics",
        "preferred_extensions": [".csv", ".tsv", ".txt", ".xlsx", ".mzTab"],
    },
    "PXD059929": {
        "description": "Cardiac proteomics (processed)",
        "preferred_extensions": [".csv", ".tsv", ".txt", ".xlsx", ".mzTab"],
    },
    "PXD060680": {
        "description": "Systemic EV proteomics",
        "preferred_extensions": [".csv", ".tsv", ".txt", ".xlsx", ".mzTab"],
    },
}

# Extensions to skip (large raw files)
SKIP_EXTENSIONS = {".raw", ".mzML", ".mzXML", ".d", ".wiff", ".scan", ".mgf"}


class PRIDEDownloader:
    """
    Downloads processed proteomics files from PRIDE Archive.

    Parses downloaded files into pandas DataFrames when possible.

    Usage
    -----
    >>> dl = PRIDEDownloader(cache_dir=Path("data/raw/pride"))
    >>> dfs = dl.download_all_as_dataframes()
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/raw/pride"),
        project_ids: Optional[List[str]] = None,
        timeout: int = 60,
        max_retries: int = 3,
        skip_raw: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        cache_dir:
            Local directory for downloaded files.
        project_ids:
            PRIDE project accessions. Defaults to all defined projects.
        timeout:
            HTTP timeout in seconds.
        max_retries:
            Retry attempts on failure.
        skip_raw:
            If True, skip large raw instrument files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.project_ids = project_ids or list(PRIDE_PROJECTS.keys())
        self.timeout = timeout
        self.max_retries = max_retries
        self.skip_raw = skip_raw

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_all(self) -> Dict[str, List[Path]]:
        """
        Download all configured PRIDE projects.

        Returns
        -------
        Dict mapping project_id -> list of local file paths.
        """
        results: Dict[str, List[Path]] = {}
        for pid in self.project_ids:
            logger.info("Processing PRIDE project: %s", pid)
            try:
                paths = self.download_project(pid)
                results[pid] = paths
            except Exception as exc:
                logger.error("Failed to download %s: %s", pid, exc)
                self._print_manual_instructions(pid)
                results[pid] = []
        return results

    def download_all_as_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Download all projects and parse files into DataFrames.

        Returns
        -------
        Dict mapping project_id -> merged DataFrame.
        """
        file_map = self.download_all()
        dfs: Dict[str, pd.DataFrame] = {}
        for pid, paths in file_map.items():
            df = self._parse_project_files(pid, paths)
            if df is not None and not df.empty:
                dfs[pid] = df
        return dfs

    def download_project(self, project_id: str) -> List[Path]:
        """
        Download all processed files for a single PRIDE project.

        Parameters
        ----------
        project_id:
            PRIDE project accession (e.g. ``PXD021371``).

        Returns
        -------
        List of local file paths.
        """
        proj_dir = self.cache_dir / project_id
        proj_dir.mkdir(parents=True, exist_ok=True)

        # Check cache
        cached = [p for p in proj_dir.iterdir() if p.is_file()]
        if cached:
            logger.info("Using %d cached files for %s", len(cached), project_id)
            return cached

        file_list = self._get_file_list(project_id)
        if not file_list:
            logger.error("No files found for %s", project_id)
            self._print_manual_instructions(project_id)
            return []

        downloaded: List[Path] = []
        prefs = PRIDE_PROJECTS.get(project_id, {}).get("preferred_extensions", [])

        # Sort: preferred extensions first
        def sort_key(item: Tuple[str, str]) -> int:
            _, name = item
            ext = Path(name).suffix.lower()
            return 0 if ext in prefs else (1 if not self.skip_raw or ext not in SKIP_EXTENSIONS else 99)

        sorted_files = sorted(file_list, key=sort_key)

        for url, filename in sorted_files:
            ext = Path(filename).suffix.lower()
            if self.skip_raw and ext in SKIP_EXTENSIONS:
                logger.debug("Skipping raw file: %s", filename)
                continue

            dest = proj_dir / filename
            if dest.exists():
                logger.info("  Cached: %s", filename)
                downloaded.append(dest)
                continue

            path = self._download_file(url, dest)
            if path:
                downloaded.append(path)

        return downloaded

    def parse_file_to_dataframe(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Parse a single downloaded file into a DataFrame.

        Parameters
        ----------
        path:
            Local path to CSV, TSV, TXT, Excel, or mzTab file.

        Returns
        -------
        DataFrame or None if parsing fails.
        """
        ext = path.suffix.lower()
        try:
            if ext in (".csv",):
                return pd.read_csv(path, low_memory=False)
            elif ext in (".tsv", ".txt"):
                return pd.read_csv(path, sep="\t", low_memory=False)
            elif ext in (".xlsx", ".xls"):
                return pd.read_excel(path)
            elif ext == ".mztab":
                return self._parse_mztab(path)
            else:
                # Attempt tab-separated as fallback
                return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", path.name, exc)
            return None

    # ------------------------------------------------------------------
    # PRIDE REST API
    # ------------------------------------------------------------------

    def _get_file_list(self, project_id: str) -> List[Tuple[str, str]]:
        """
        Retrieve file list from PRIDE REST API.

        Returns list of (url, filename) tuples.
        """
        files: List[Tuple[str, str]] = []

        # v2 API endpoint
        url = f"{PRIDE_API_BASE}/files/byProject?accession={project_id}&pageSize=200&page=0"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "PRIDEDownloader/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())

            for item in data:
                file_url = item.get("publicFileLocations", [{}])[0].get("value", "")
                if not file_url:
                    # Try FTP fallback
                    file_url = item.get("fileName", "")
                fname = item.get("fileName", Path(file_url).name if file_url else "unknown")
                if file_url and fname:
                    files.append((file_url, fname))

        except Exception as exc:
            logger.warning("PRIDE REST API failed for %s: %s", project_id, exc)
            files = self._get_file_list_ftp_fallback(project_id)

        return files

    def _get_file_list_ftp_fallback(self, project_id: str) -> List[Tuple[str, str]]:
        """
        Fallback: list files via PRIDE FTP HTTP index.

        Returns list of (url, filename) tuples.
        """
        import re

        # Derive year from project submission (heuristic based on accession number)
        # PXD accessions don't encode year directly; use the PRIDE search API
        search_url = (
            f"{PRIDE_API_BASE}/projects/{project_id}"
        )
        year = "unknown"
        try:
            req = urllib.request.Request(
                search_url,
                headers={"Accept": "application/json", "User-Agent": "PRIDEDownloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                meta = json.loads(resp.read().decode())
            submission_date = meta.get("submissionDate", "") or meta.get("publicationDate", "")
            if submission_date:
                year = submission_date[:4]
        except Exception:
            pass

        ftp_url = f"{PRIDE_FTP_BASE}/{year}/{project_id}/"
        files: List[Tuple[str, str]] = []
        try:
            req = urllib.request.Request(
                ftp_url,
                headers={"User-Agent": "PRIDEDownloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            for m in re.finditer(r'href="([^"/][^"]*)"', html):
                name = m.group(1)
                files.append((ftp_url + name, name))
        except Exception as exc:
            logger.debug("FTP fallback failed for %s: %s", project_id, exc)

        return files

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_file(
        self, url: str, dest: Path, chunk_size: int = 512 * 1024
    ) -> Optional[Path]:
        """
        Download a single file with progress and retries.

        Returns dest on success, None on failure.
        """
        filename = dest.name
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "PRIDEDownloader/1.0"},
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
                    finally:
                        if bar:
                            bar.close()

                tmp.rename(dest)
                logger.info("  Downloaded: %s", filename)
                return dest

            except (URLError, OSError) as exc:
                logger.warning(
                    "  Attempt %d/%d failed for %s: %s", attempt, self.max_retries, filename, exc
                )
                if tmp.exists():
                    tmp.unlink()
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("  All download attempts failed for %s", filename)
        return None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_project_files(self, project_id: str, paths: List[Path]) -> Optional[pd.DataFrame]:
        """
        Parse and merge all downloaded files for a project into one DataFrame.

        Merges on protein/gene identifiers when possible.
        """
        frames: List[pd.DataFrame] = []
        for path in paths:
            df = self.parse_file_to_dataframe(path)
            if df is not None and not df.empty:
                df["_source_file"] = path.name
                df["_project"] = project_id
                frames.append(df)

        if not frames:
            return None

        if len(frames) == 1:
            return frames[0]

        # Attempt merge on common protein columns
        common_cols = ["Accession", "Gene", "Protein", "ProteinID", "UniProtKB"]
        for col in common_cols:
            matching = [f for f in frames if col in f.columns]
            if len(matching) == len(frames):
                logger.info("Merging %d DataFrames on column '%s'", len(frames), col)
                merged = frames[0]
                for other in frames[1:]:
                    try:
                        merged = pd.merge(merged, other, on=col, how="outer", suffixes=("", f"_{other['_source_file'].iloc[0][:8]}"))
                    except Exception:
                        pass
                return merged

        # No common merge column: concatenate
        return pd.concat(frames, axis=0, ignore_index=True)

    def _parse_mztab(self, path: Path) -> pd.DataFrame:
        """
        Parse an mzTab file, extracting the protein quantification section.

        Parameters
        ----------
        path:
            Local mzTab file path.

        Returns
        -------
        DataFrame with protein rows.
        """
        rows = []
        in_prt_section = False
        header: Optional[List[str]] = None

        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("PRH"):
                    header = line.split("\t")
                    in_prt_section = True
                elif in_prt_section and line.startswith("PRT"):
                    if header:
                        parts = line.split("\t")
                        rows.append(dict(zip(header, parts)))
                elif in_prt_section and not line.startswith("PRT"):
                    in_prt_section = False

        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Manual instructions
    # ------------------------------------------------------------------

    def _print_manual_instructions(self, project_id: str) -> None:
        """Print manual download instructions for a PRIDE project."""
        info = PRIDE_PROJECTS.get(project_id, {})
        dest = self.cache_dir / project_id
        print(
            f"\n{'='*60}\n"
            f"MANUAL DOWNLOAD REQUIRED: {project_id}\n"
            f"Description: {info.get('description', '')}\n"
            f"{'='*60}\n"
            f"1. Go to: https://www.ebi.ac.uk/pride/archive/projects/{project_id}\n"
            f"2. Download processed result files (.csv/.tsv/.mzTab)\n"
            f"3. Avoid downloading raw .raw/.mzML files\n"
            f"4. Place files in: {dest.absolute()}\n"
            f"{'='*60}\n"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_downloaded(self) -> Dict[str, List[Path]]:
        """Return dict of project_id -> list of local file paths."""
        return {
            pid: sorted(
                p for p in (self.cache_dir / pid).iterdir() if p.is_file()
            )
            if (self.cache_dir / pid).exists()
            else []
            for pid in self.project_ids
        }
