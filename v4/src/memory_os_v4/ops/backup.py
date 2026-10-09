"""Encrypted backups and restore drills for the memory-os-v4 store (runbook RB-04).

Until the transactional SQLite store (ADR-001) provides an online backup API, a *consistent* backup requires the
server to be stopped: with ``MEMORY_V4_SERVICE_LABEL`` set (macOS launchd), ``backup(stop_service=True)`` boots the
service out, copies the store, and always boots it back in. Archives: tar.gz encrypted with AES-256-GCM
(``pip install 'memory-os-ai[ops]'``), a 32-byte key in an owner-only file, a manifest with SHA-256 of every file
and live item counts, and a rotation keeping the last 7 daily plus 4 weekly (Sunday) archives.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from .auth import health_ok
from .settings import OpsSettings

MAGIC = b"MOSB1"  # MAGIC | nonce(12) | AESGCM(tar.gz, aad=MAGIC)
EXCLUDE = ("server.log", "*.tmp")
ARCHIVE_GLOB = "memory-os-v4-*.mosb"


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # optional dependency
        raise RuntimeError("Encrypted backups need: pip install 'memory-os-ai[ops]'") from exc
    return AESGCM(key)


def load_or_create_key(path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(os.urandom(32))
    key = path.read_bytes()
    if len(key) != 32 or (os.stat(path).st_mode & 0o077):
        raise ValueError(f"{path} must hold 32 bytes with mode 0600")
    return key


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def item_counts(items: Path) -> dict:
    last: dict[str, dict] = {}
    malformed = 0
    if items.exists():
        with items.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    last[d["id"]] = d
                except (ValueError, KeyError):
                    malformed += 1
    by_kind: dict[str, int] = {}
    for d in last.values():
        if not d.get("deleted"):
            by_kind[d.get("kind", "?")] = by_kind.get(d.get("kind", "?"), 0) + 1
    return {
        "live_by_kind": by_kind,
        "live_total": sum(by_kind.values()),
        "ids": len(last),
        "malformed_lines": malformed,
    }


def encrypt(data: bytes, key: bytes) -> bytes:
    nonce = os.urandom(12)
    return MAGIC + nonce + _aesgcm(key).encrypt(nonce, data, MAGIC)


def decrypt(blob: bytes, key: bytes) -> bytes:
    if blob[:5] != MAGIC:
        raise ValueError("not a memory-os backup archive")
    return _aesgcm(key).decrypt(blob[5:17], blob[17:], MAGIC)


def rotate(dest: Path, daily: int = 7, weekly: int = 4) -> list[str]:
    files = sorted(dest.glob(ARCHIVE_GLOB))
    keep = set(files[-daily:])
    sundays = [f for f in files if dt.datetime.strptime(f.stem.split("-v4-")[1][:8], "%Y%m%d").weekday() == 6]
    keep |= set(sundays[-weekly:])
    removed = []
    for f in files:
        if f not in keep:
            f.unlink()
            removed.append(f.name)
    return removed


def _launchctl(*args: str) -> None:
    subprocess.run(["launchctl", *args], capture_output=True, check=False)


def backup(
    settings: OpsSettings | None = None, *, stop_service: bool = True, now: dt.datetime | None = None
) -> dict:
    settings = settings or OpsSettings.from_env()
    key = load_or_create_key(settings.backup_key)
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    plist = settings.launchd_plist if stop_service else None
    if stop_service and plist is None:
        raise RuntimeError("A consistent backup needs MEMORY_V4_SERVICE_LABEL (or use stop_service=False)")
    record: dict = {"stamp": stamp, "consistent_server_stopped": bool(plist), "ok": False}
    with tempfile.TemporaryDirectory(prefix="mosb-") as tmp:
        staging = Path(tmp) / "memory-os-v4"
        domain = f"gui/{os.getuid()}"
        try:
            if plist:
                _launchctl("bootout", f"{domain}/{settings.service_label}")
                time.sleep(2)
            shutil.copytree(settings.store_dir, staging, ignore=shutil.ignore_patterns(*EXCLUDE))
        finally:
            if plist:
                _launchctl("bootstrap", domain, str(plist))
        if plist:
            deadline = time.time() + 90
            while time.time() < deadline and not health_ok(settings):
                time.sleep(2)
            record["server_restarted_healthy"] = health_ok(settings)
        files = sorted(p for p in staging.rglob("*") if p.is_file())
        manifest = {
            "format": "memory-os-v4-backup/1",
            "created": stamp,
            "consistent_server_stopped": bool(plist),
            "files": {
                str(p.relative_to(staging)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files
            },
            "items": item_counts(staging / "items.jsonl"),
        }
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(staging, arcname="memory-os-v4")
        blob = encrypt(buf.getvalue(), key)
    settings.backup_dir.mkdir(parents=True, exist_ok=True)
    name = f"memory-os-v4-{stamp}.mosb"
    part = settings.backup_dir / (name + ".part")
    part.write_bytes(blob)
    os.chmod(part, 0o600)
    part.rename(settings.backup_dir / name)
    record.update(
        archive=name, bytes=len(blob), items=manifest["items"], rotated_out=rotate(settings.backup_dir)
    )
    record["ok"] = record.get("server_restarted_healthy", True) is not False
    _log(settings, "backup.jsonl", record)
    return record


def extract(archive: Path, key: bytes, target: Path) -> Path:
    with tarfile.open(fileobj=io.BytesIO(decrypt(archive.read_bytes(), key)), mode="r:gz") as tar:
        tar.extractall(target, filter="data")
    return target / "memory-os-v4"


def verify_manifest(root: Path) -> tuple[dict, list[str]]:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    bad = [
        f
        for f, m in manifest["files"].items()
        if not (root / f).is_file()
        or (root / f).stat().st_size != m["bytes"]
        or sha256(root / f) != m["sha256"]
    ]
    return manifest, bad


def restore_drill(
    settings: OpsSettings | None = None,
    archive: Path | None = None,
    *,
    start_server: bool = True,
    port: int = 8799,
) -> dict:
    """Decrypt into a throw-away directory, verify the manifest, optionally serve it on a temporary port and
    compare item counts. Never touches the live store."""
    settings = settings or OpsSettings.from_env()
    t0 = time.time()
    archive = archive or sorted(settings.backup_dir.glob(ARCHIVE_GLOB))[-1]
    record: dict = {"archive": archive.name, "ok": False}
    with tempfile.TemporaryDirectory(prefix="mosr-") as tmp:
        root = extract(archive, settings.backup_key.read_bytes(), Path(tmp))
        manifest, bad = verify_manifest(root)
        created = dt.datetime.strptime(manifest["created"], "%Y%m%d-%H%M%S")
        record.update(
            manifest_files=len(manifest["files"]),
            manifest_mismatches=bad,
            backup_age_hours=round((dt.datetime.now() - created).total_seconds() / 3600, 2),
            expected_live_total=manifest["items"]["live_total"],
        )
        if start_server and not bad:
            record["server_check"] = _serve_and_count(root, settings, port, manifest["items"]["live_total"])
        record["ok"] = not bad and (not start_server or record.get("server_check", {}).get("match") is True)
    record["duration_s"] = round(time.time() - t0, 1)
    _log(settings, "restore_drill.jsonl", record)
    return record


def _serve_and_count(root: Path, settings: OpsSettings, port: int, expected: int) -> dict:
    import sys

    from .auth import call_tool

    env = dict(
        os.environ,
        MEMORY_V4_DIR=str(root),
        MEMORY_V4_PROJECT="*",
        MEMORY_V4_TOKEN_FILE=str(settings.token_file),
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "memory_os_v4.server",
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp_settings = OpsSettings(**{**settings.__dict__, "url": f"http://127.0.0.1:{port}/mcp/"})
    try:
        import anyio

        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                status = anyio.run(call_tool, "memory_status", {}, tmp_settings)
                return {
                    "status_total": status["total"],
                    "expected_total": expected,
                    "match": status["total"] == expected,
                }
            except Exception:
                time.sleep(3)
        return {"error": "temporary server did not answer"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def _log(settings: OpsSettings, name: str, record: dict) -> None:
    logs = settings.ops_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / name).open("a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
