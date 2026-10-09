"""Download the two bench datasets into bench/data/.

Usage:
    PYTHONPATH=src:. .venv/bin/python -m bench.download [--dataset locomo|longmemeval_oracle|all] [--force]

Only ever fetches the two small files explicitly authorised for this bench:
  - LoCoMo10        (~2.8 MB)  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
  - LongMemEval oracle (~15.4 MB) https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle

Does NOT fetch longmemeval_s (278 MB) or longmemeval_m (multi-GB) — do not add them here.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

USER_AGENT = "memory-os-v4-bench/1.0 (+local eval; contact: local)"


@dataclass
class DatasetSpec:
    key: str
    url: str
    filename: str


DATASETS = {
    "locomo": DatasetSpec(
        key="locomo",
        url="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        filename="locomo10.json",
    ),
    "longmemeval_oracle": DatasetSpec(
        key="longmemeval_oracle",
        url="https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle",
        filename="longmemeval_oracle.json",
    ),
}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_one(spec: DatasetSpec, force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / spec.filename
    if dest.exists() and not force:
        size = dest.stat().st_size
        digest = sha256_of(dest)
        print(f"[skip] {spec.key}: {dest} already present ({human_size(size)}, sha256={digest})")
        return dest

    print(f"[get ] {spec.key}: {spec.url}")
    req = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        read = 0
        chunk_size = 1 << 20
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r  {human_size(read)} / {human_size(total)} ({pct:5.1f}%)", end="", flush=True)
        print()
    tmp.rename(dest)

    size = dest.stat().st_size
    digest = sha256_of(dest)
    print(f"[done] {spec.key}: {dest} ({human_size(size)}, sha256={digest})")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["locomo", "longmemeval_oracle", "all"], default="all")
    p.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = p.parse_args(argv)

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for key in keys:
        try:
            download_one(DATASETS[key], force=args.force)
        except Exception as exc:  # pragma: no cover - network path
            print(f"[FAIL] {key}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
