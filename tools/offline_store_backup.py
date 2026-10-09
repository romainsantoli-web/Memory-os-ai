"""Encrypted, verified offline snapshot. Never starts/stops a service or extracts files.

Compatible MOSB1 envelope; use only with all writers stopped. --verify is read-only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"MOSB1"


def load_key(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid() or info.st_nlink != 1):
            raise ValueError("Backup key must be an owner-only regular 0600 file")
        key = fh.read(33)
    if len(key) != 32:
        raise ValueError("Backup key must have exactly 32 bytes")
    return key


def snapshot(source: Path) -> dict[str, bytes]:
    source = source.resolve(strict=True)
    paths = sorted(source.rglob("*"))
    if any(p.is_symlink() or (not p.is_dir() and not p.is_file()) for p in paths):
        raise ValueError("Linked/special paths are forbidden in backup source")
    return {str(p.relative_to(source)): p.read_bytes() for p in paths if p.is_file()}


def inventory(files: dict[str, bytes]) -> dict:
    return {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in files.items()}


def item_counts(data: bytes) -> dict:
    if data and not data.endswith(b"\n"):
        raise ValueError("Truncated items log")
    latest = {r["id"]: r for r in (json.loads(line) for line in data.splitlines() if line.strip())}
    counts = {}
    for row in latest.values():
        if not row.get("deleted"):
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return {"live_by_kind": counts, "live_total": sum(counts.values()),
            "ids": len(latest), "malformed_lines": 0}


def backup(source: Path, archive: Path, key_file: Path, *, offline: bool = False) -> dict:
    if not offline:
        raise ValueError("Explicit --offline acknowledgement is required")
    source = source.resolve(strict=True)
    if source in archive.resolve().parents or archive.exists():
        raise ValueError("Archive must be new and outside the source")
    key = load_key(key_file)
    files = snapshot(source)
    if "MANIFEST.json" in files:
        raise ValueError("A restored snapshot with an existing MANIFEST requires review")
    manifest = {"format": "memory-os-v4-backup/1", "created": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                "consistent_server_stopped": True, "offline_acknowledged": True,
                "files": inventory(files), "items": item_counts(files["items.jsonl"])}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in {**files, "MANIFEST.json": (json.dumps(manifest) + "\n").encode()}.items():
            entry = tarfile.TarInfo("memory-os-v4/" + name)
            entry.size, entry.mode = len(data), 0o600
            tar.addfile(entry, io.BytesIO(data))
    if inventory(snapshot(source)) != manifest["files"]:
        raise ValueError("Source changed during offline backup")
    nonce = os.urandom(12)
    blob = MAGIC + nonce + AESGCM(key).encrypt(nonce, buf.getvalue(), MAGIC)
    temporary = archive.with_name(archive.name + ".part")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    # Exclusive publication: never overwrite another backup.
    os.link(temporary, archive)
    temporary.unlink()
    fd = os.open(archive.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return verify(archive, key_file, source=source)


def verify(archive: Path, key_file: Path, *, source: Path | None = None) -> dict:
    blob = archive.read_bytes()
    if blob[:5] != MAGIC:
        raise ValueError("Unknown backup format")
    data = AESGCM(load_key(key_file)).decrypt(blob[5:17], blob[17:], MAGIC)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        files = {}
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if (not member.isfile() or not parts or parts[0] != "memory-os-v4"
                    or ".." in parts or len(parts) < 2):
                raise ValueError("Unsafe archive member")
            name = str(Path(*parts[1:]))
            if name in files:
                raise ValueError("Duplicate archive member")
            files[name] = tar.extractfile(member).read()
    manifest = json.loads(files.pop("MANIFEST.json"))
    if inventory(files) != manifest["files"] or item_counts(files["items.jsonl"]) != manifest["items"]:
        raise ValueError("Backup manifest/data mismatch")
    if source is not None and inventory(snapshot(source)) != manifest["files"]:
        raise ValueError("Frozen source differs from encrypted backup")
    return {"verified": True, "files": len(files), "items": manifest["items"],
            "consistent_server_stopped": manifest["consistent_server_stopped"],
            "archive_sha256": hashlib.sha256(blob).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and args.source is None:
        parser.error("--source is required when creating a backup")
    result = (verify(args.archive, args.key_file, source=args.source) if args.verify
              else backup(args.source, args.archive, args.key_file, offline=args.offline))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
