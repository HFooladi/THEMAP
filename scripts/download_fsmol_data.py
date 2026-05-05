#!/usr/bin/env python
"""Download and extract the FS-Mol hardness dataset from Zenodo.

Reproduces the data setup for the paper:
    Fooladi, Hirte, Kirchmair (2024).
    "Quantifying the hardness of bioactivity prediction tasks for transfer learning."
    J. Chem. Inf. Model. 64(10), 4031-4046.
    https://doi.org/10.5281/zenodo.10605093

Usage:
    python scripts/download_fsmol_data.py
    python scripts/download_fsmol_data.py --dest /tmp/fsmol --keep-zip
    make download-fsmol

The archive is ~16.2 GB; ensure ~35 GB of free disk space for download + extract.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "10605093"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/fsmol_hardness.zip"
EXPECTED_MD5 = "10644660a53d8d106b6883cb53eb1f3b"
EXPECTED_BYTES = 16_165_999_857
ARCHIVE_NAME = "fsmol_hardness.zip"
DEFAULT_DEST = Path("datasets") / "fsmol_hardness"
CHUNK = 1024 * 1024  # 1 MiB
MAX_RETRIES = 5


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def _progress(downloaded: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    speed = downloaded / elapsed
    pct = (downloaded / total * 100) if total else 0.0
    eta_s = (total - downloaded) / speed if speed > 0 and total else 0
    eta = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s" if total else "?"
    sys.stderr.write(
        f"\r  {_human(downloaded)} / {_human(total)} ({pct:5.1f}%) at {_human(speed)}/s  ETA {eta}   "
    )
    sys.stderr.flush()


def _download(url: str, dest: Path, expected_size: int) -> None:
    """Stream-download `url` to `dest` with resume + retries."""
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        existing = dest.stat().st_size if dest.exists() else 0
        if existing >= expected_size:
            return
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Server may ignore Range and start at 0 — handle both cases.
                total = expected_size
                if existing and resp.status != 206:
                    existing = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing else "wb"
                with dest.open(mode) as f:
                    downloaded = existing
                    while True:
                        buf = resp.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        _progress(downloaded, total, started)
            sys.stderr.write("\n")
            if dest.stat().st_size >= expected_size:
                return
            raise IOError(f"Short download: got {dest.stat().st_size} bytes, expected {expected_size}")
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            sys.stderr.write(f"\n  Attempt {attempt} failed: {e}\n")
            if attempt >= MAX_RETRIES:
                raise
            backoff = min(2**attempt, 30)
            sys.stderr.write(f"  Retrying in {backoff}s (resuming)...\n")
            time.sleep(backoff)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    size = path.stat().st_size
    read = 0
    started = time.monotonic()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
            read += len(chunk)
            _progress(read, size, started)
    sys.stderr.write("\n")
    return h.hexdigest()


def _is_extracted(target: Path) -> bool:
    """Heuristic: dataset is already extracted if any expected subdir exists."""
    if not target.is_dir():
        return False
    expected = ("ext_chem", "ext_prot", "int_chem", "FSMol_Eval_ProtoNet")
    return any((target / name).exists() for name in expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract the FS-Mol hardness dataset from Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Target directory for extracted data (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded ZIP after extraction (default: delete to save disk space)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if data already exists",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 verification (not recommended)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download only; do not extract",
    )
    args = parser.parse_args(argv)

    dest: Path = args.dest.resolve()

    if not args.force and _is_extracted(dest):
        print(f"Dataset already extracted at {dest}. Use --force to re-download.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / ARCHIVE_NAME

    print(f"Downloading {ZENODO_URL}")
    print(f"  -> {archive}")
    print(f"  Expected size: {_human(EXPECTED_BYTES)} (md5 {EXPECTED_MD5})")
    if archive.exists() and archive.stat().st_size >= EXPECTED_BYTES and not args.force:
        print("  Archive already present at expected size; skipping download.")
    else:
        if args.force and archive.exists():
            archive.unlink()
        _download(ZENODO_URL, archive, EXPECTED_BYTES)
        print(f"Downloaded {_human(archive.stat().st_size)}.")

    if not args.no_verify:
        print("Verifying MD5...")
        actual = _md5(archive)
        if actual != EXPECTED_MD5:
            print(
                f"ERROR: MD5 mismatch.\n  expected: {EXPECTED_MD5}\n  actual:   {actual}",
                file=sys.stderr,
            )
            return 2
        print("MD5 OK.")

    if args.no_extract:
        print(f"Skipping extraction (--no-extract). Archive: {archive}")
        return 0

    print(f"Extracting to {dest}...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    print("Extraction complete.")

    # Some Zenodo archives contain a single top-level directory; flatten if so.
    entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and entries[0].name == "fsmol_hardness":
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), str(dest / child.name))
        inner.rmdir()
        print("Flattened nested fsmol_hardness/ directory.")

    if not args.keep_zip:
        archive.unlink()
        print(f"Removed archive {archive} (use --keep-zip to keep it).")

    print(f"\nDone. FS-Mol hardness data is available at: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
