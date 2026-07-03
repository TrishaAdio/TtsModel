#!/usr/bin/env python3
"""
download_dataset.py
-------------------
Download the source voice clips listed in voices.txt into data/raw/.
Colored progress via colorama.

Usage:
  python src/download_dataset.py                       # uses ./voices.txt -> ./data/raw
  python src/download_dataset.py --list voices.txt --out data/raw
  python src/download_dataset.py --force               # re-download even if present
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

# allow running as "python src/download_dataset.py" or "python -m ..."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from console import banner, step, info, ok, warn, err, value, progress_bar  # noqa: E402

CHUNK = 64 * 1024


def read_urls(list_path: Path) -> list[str]:
    urls = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def filename_for(url: str, index: int) -> str:
    """Derive a stable filename from the URL, prefixed with an index for ordering."""
    name = os.path.basename(urlparse(url).path) or f"clip_{index:03d}"
    stem, ext = os.path.splitext(name)
    ext = ext or ".ogg"
    # short hash keeps names unique even if basenames collide
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{index:03d}_{h}{ext}"


def download_one(url: str, dest: Path, force: bool) -> bool:
    if dest.exists() and not force and dest.stat().st_size > 0:
        info(f"skip (exists): {dest.name}")
        return True
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                err(f"HTTP {r.status_code} for {url}")
                return False
            total = int(r.headers.get("Content-Length", 0))
            tmp = dest.with_suffix(dest.suffix + ".part")
            got = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        progress_bar(got, total, prefix=dest.name)
            if total:
                progress_bar(total, total, prefix=dest.name)
            else:
                info(f"downloaded {dest.name} ({got} bytes)")
            tmp.replace(dest)
        return True
    except requests.RequestException as e:
        err(f"failed {url}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Download voice clips listed in a URL file.")
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--list", default=str(here / "voices.txt"), help="Path to URL list.")
    ap.add_argument("--out", default=str(here / "data" / "raw"), help="Output directory.")
    ap.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = ap.parse_args()

    banner("Voice dataset downloader")

    list_path = Path(args.list)
    if not list_path.is_file():
        err(f"URL list not found: {list_path}")
        return 1

    urls = read_urls(list_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    value("URL list", list_path)
    value("Output dir", out)
    value("Clips", len(urls))
    print()

    success = 0
    for i, url in enumerate(urls, 1):
        step(i, len(urls), url)
        dest = out / filename_for(url, i)
        if download_one(url, dest, args.force):
            success += 1

    print()
    if success == len(urls):
        ok(f"All {success} clips downloaded to {out}")
    else:
        warn(f"{success}/{len(urls)} clips downloaded. Check errors above.")
    info("Next: python src/prepare_dataset.py --input-dir data/raw --output data/dataset/speaker1")
    return 0 if success == len(urls) else 2


if __name__ == "__main__":
    raise SystemExit(main())
