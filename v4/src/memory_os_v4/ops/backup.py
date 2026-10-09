"""Encrypted backups and restore drills for the JSONL memory-os-v4 store (runbook RB-04).

A backup is *qualified* only with positive evidence that no writer can touch the store during the copy:
either the store's ``writer.lock`` is held exclusively by this process, or the launchd service was loaded, booted
out by this operation, and is verifiably gone (no launchd job, no ``memory_os_v4`` server process, port closed).
Otherwise the operation refuses (use the attested maintenance tooling) — a ``--no-stop`` copy is only a labelled,
non-qualified *diagnostic snapshot*. Only a service stopped by this operation is restarted.

Archives: strict snapshot (no links or special files, strict JSONL), tar of regular files only, AES-256-GCM
(``pip install 'memory-os-ai[ops]'``), exclusive 0600 publication without overwrite, fsync, full read-back
verification (decrypt, safe extract, exact inventory, hashes, item counts) BEFORE rotation. Rotation keeps the
newest archive of the 7 most recent distinct days and of the 4 most recent distinct ISO weeks. SQLite stores are
refused until the online backup API (ADR-001 S4).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import socket
import stat
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from .auth import SecretFileError, read_secret_file, write_secret_file
from .settings import OpsSettings

MAGIC = b"MOSB1"  # MAGIC | nonce(12) | AESGCM(tar.gz, aad=MAGIC)
ROOT = "memory-os-v4"
ARCHIVE_GLOB = "memory-os-v4-*.mosb"
EXCLUDED = {"server.log"}
SQLITE_MARKERS = (".sqlite", ".sqlite3", ".db", "-wal", "-shm")


class BackupError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- crypto / key
def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # optional dependency
        raise BackupError("Encrypted backups need: pip install 'memory-os-ai[ops]'") from exc
    return AESGCM(key)


def load_key(path: Path) -> bytes:
    """Strict read of an existing key (owner-only regular 0600 file, one link, exactly 32 bytes)."""
    try:
        return read_secret_file(path, max_bytes=32, exact=32)
    except FileNotFoundError:
        raise SecretFileError(f"{path}: backup key missing (never recreated to decrypt)") from None


def load_or_create_key(path: Path) -> bytes:
    """Backups only: create the key exclusively when absent, otherwise validate it strictly."""
    if not path.exists() and not path.is_symlink():
        write_secret_file(path, os.urandom(32))
    return load_key(path)


def encrypt(data: bytes, key: bytes) -> bytes:
    nonce = os.urandom(12)
    return MAGIC + nonce + _aesgcm(key).encrypt(nonce, data, MAGIC)


def decrypt(blob: bytes, key: bytes) -> bytes:
    if blob[:5] != MAGIC or len(blob) < 17 + 16:
        raise ValueError("not a memory-os backup archive")
    return _aesgcm(key).decrypt(blob[5:17], blob[17:], MAGIC)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- snapshot
def parse_items(data: bytes) -> dict:
    """Strict JSONL: every line a JSON object with a string ``id``; non-empty data ends with a newline."""
    if data and not data.endswith(b"\n"):
        raise BackupError("items.jsonl does not end with a newline (truncated write: result uncertain)")
    last: dict[str, dict] = {}
    for n, line in enumerate(data.splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError:
            raise BackupError(f"items.jsonl line {n} is not valid JSON") from None
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise BackupError(f"items.jsonl line {n} has no string id")
        last[row["id"]] = row
    by_kind: dict[str, int] = {}
    for row in last.values():
        if not row.get("deleted"):
            by_kind[row.get("kind", "?")] = by_kind.get(row.get("kind", "?"), 0) + 1
    return {"live_by_kind": by_kind, "live_total": sum(by_kind.values()), "ids": len(last)}


def snapshot(store: Path) -> dict[str, bytes]:
    store = store.resolve(strict=True)
    files: dict[str, bytes] = {}
    for path in sorted(store.rglob("*")):
        rel = path.relative_to(store).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise BackupError(f"link or special file refused in store: {rel}")
        if rel.endswith(SQLITE_MARKERS):
            raise BackupError("SQLite store: use the online backup API (ADR-001 S4), not a file copy")
        if stat.S_ISREG(mode) and path.name not in EXCLUDED and not rel.endswith(".tmp"):
            files[rel] = path.read_bytes()
    if "items.jsonl" not in files:
        raise BackupError("items.jsonl missing from store")
    return files


def build_archive(files: dict[str, bytes], manifest: dict) -> bytes:
    payload = dict(files)
    payload["MANIFEST.json"] = json.dumps(manifest, indent=1, ensure_ascii=False).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in sorted(payload):
            info = tarfile.TarInfo(f"{ROOT}/{rel}")
            info.size, info.mode, info.mtime = len(payload[rel]), 0o600, 0
            tar.addfile(info, io.BytesIO(payload[rel]))
    return buf.getvalue()


def extract(archive: Path, key: bytes, target: Path) -> Path:
    """Decrypt and extract safely: regular files and directories only, relative paths under ``memory-os-v4/``,
    no traversal, no duplicates."""
    with tarfile.open(fileobj=io.BytesIO(decrypt(archive.read_bytes(), key)), mode="r:gz") as tar:
        members, seen = tar.getmembers(), set()
        for m in members:
            p = PurePosixPath(m.name)
            if not (m.isfile() or m.isdir()) or p.is_absolute() or ".." in p.parts:
                raise BackupError(f"unsafe archive member: {m.name!r}")
            if p.parts[0] != ROOT or m.name in seen:
                raise BackupError(f"unexpected or duplicate archive member: {m.name!r}")
            seen.add(m.name)
        tar.extractall(target, members=members, filter="data")
    return target / ROOT


def verify_manifest(root: Path) -> tuple[dict, list[str]]:
    """Exact inventory: every manifested file present with its size and hash, no extra file, item counts match."""
    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected = manifest["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    actual.discard("MANIFEST.json")
    problems = sorted(f"extra:{f}" for f in actual - set(expected))
    for rel, meta in expected.items():
        p = root / rel
        if p.is_symlink() or not p.is_file():
            problems.append(f"missing:{rel}")
        elif p.stat().st_size != meta["bytes"] or sha256(p) != meta["sha256"]:
            problems.append(f"altered:{rel}")
    if not problems:
        try:
            counts = parse_items((root / "items.jsonl").read_bytes())
        except BackupError as exc:
            problems.append(f"items:{exc}")
        else:
            if counts != manifest["items"]:
                problems.append("items:counts differ from manifest")
    return manifest, problems


# ---------------------------------------------------------------------------------------------- service control
def _launchctl(*args: str) -> int:
    return subprocess.run(["launchctl", *args], capture_output=True, check=False).returncode


def _service_loaded(label: str) -> bool:
    return _launchctl("print", f"gui/{os.getuid()}/{label}") == 0


def _writer_processes() -> list[str]:
    out = subprocess.run(["pgrep", "-fl", "memory_os_v4.server|memory-os-(ai|v4) .*--transport"],
                         capture_output=True, text=True, check=False).stdout
    return [line for line in out.splitlines() if line.strip() and str(os.getpid()) != line.split()[0]]


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_stopped(settings: OpsSettings, label: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _service_loaded(label) and not _writer_processes() and not _port_open(settings.port):
            return True
        time.sleep(1)
    return False


def _try_writer_lock(store: Path):
    """``None`` when the store has no writer.lock (4.0.0 engines), ``"busy"`` when another process holds it,
    else an open file holding an exclusive lock (candidate engines hold it while open)."""
    lock_path = store / "writer.lock"
    if not lock_path.exists() and not lock_path.is_symlink():
        return None
    if lock_path.is_symlink() or not lock_path.is_file():
        raise BackupError("writer.lock is not a regular file")
    import fcntl

    fh = lock_path.open("r+b")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return "busy"
    return fh


# ---------------------------------------------------------------------------------------------- backup
def backup(settings: OpsSettings | None = None, *, stop_service: bool = True, now: dt.datetime | None = None) -> dict:
    settings = settings or OpsSettings.from_env()
    key = load_or_create_key(settings.backup_key)
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    record: dict = {"stamp": stamp, "qualified": False, "verified": False, "ok": False}
    stopped_by_us = False
    lock = None
    try:
        files = None
        if stop_service:
            label, plist = settings.service_label, settings.launchd_plist
            if not label or plist is None:
                raise BackupError("a qualified backup needs MEMORY_V4_SERVICE_LABEL")
            lock = _try_writer_lock(settings.store_dir)
            if lock is None or lock == "busy":  # need a verified stop of the managed service
                if not _service_loaded(label):
                    raise BackupError("service not loaded and no free writer.lock: cannot prove that no writer is "
                                      "running; use the attested offline maintenance backup")
                if _launchctl("bootout", f"gui/{os.getuid()}/{label}") != 0:
                    raise BackupError("launchctl bootout failed")
                stopped_by_us = True
                if not _wait_stopped(settings, label):
                    raise BackupError("service, writer process or port still alive after bootout")
                if lock == "busy":
                    lock = _try_writer_lock(settings.store_dir)
                    if lock in (None, "busy"):
                        raise BackupError("writer.lock still held after the service stopped")
            record["evidence"] = "writer_lock" if lock else "verified_bootout"
            files = snapshot(settings.store_dir)
            record["qualified"] = True
        else:
            files = snapshot(settings.store_dir)
            record["evidence"] = "none (diagnostic snapshot, writers may be running)"
        items = parse_items(files["items.jsonl"])
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        items = None
    finally:
        if lock not in (None, "busy"):
            lock.close()
        if stopped_by_us:
            record["restart_rc"] = _launchctl("bootstrap", f"gui/{os.getuid()}", str(settings.launchd_plist))
            record["server_restarted_healthy"] = _wait_healthy(settings)
    if items is None:
        _log(settings, "backup.jsonl", record)
        return record
    manifest = {"format": "memory-os-v4-backup/2", "created": stamp, "qualified": record["qualified"],
                "evidence": record["evidence"],
                "files": {rel: {"sha256": sha256_bytes(b), "bytes": len(b)} for rel, b in files.items()},
                "items": items}
    blob = encrypt(build_archive(files, manifest), key)
    name = f"memory-os-v4-{stamp}{'' if record['qualified'] else '-diagnostic'}.mosb"
    try:
        final = _publish(settings.backup_dir, name, blob)
        record.update(archive=name, bytes=len(blob), archive_sha256=sha256_bytes(blob), items=items)
        with tempfile.TemporaryDirectory(prefix="mosv-") as tmp:
            _, problems = verify_manifest(extract(final, key, Path(tmp)))
        record["verified"] = not problems
        record["verify_problems"] = problems
        if record["verified"] and record["qualified"]:
            record["rotated_out"] = rotate(settings.backup_dir)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["ok"] = bool(record["verified"] and record.get("server_restarted_healthy", True) is not False)
    _log(settings, "backup.jsonl", record)
    return record


def _publish(dest: Path, name: str, blob: bytes) -> Path:
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    final, part = dest / name, dest / (name + ".part")
    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, blob)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(part, final)  # fails if the name exists: never overwrite an archive
    finally:
        part.unlink()
    dfd = os.open(dest, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return final


def _wait_healthy(settings: OpsSettings, timeout: float = 90.0) -> bool:
    from .auth import health_ok

    deadline = time.time() + timeout
    while time.time() < deadline:
        if health_ok(settings):
            return True
        time.sleep(2)
    return False


def _archive_day(path: Path) -> dt.date | None:
    try:
        return dt.datetime.strptime(path.name[len("memory-os-v4-"):][:8], "%Y%m%d").date()
    except ValueError:
        return None


def rotate(dest: Path, days: int = 7, weeks: int = 4) -> list[str]:
    """Keep the newest qualified archive of the ``days`` most recent distinct days and of the ``weeks`` most
    recent distinct ISO weeks; never touch diagnostic snapshots or unparsable names."""
    archives = sorted(p for p in dest.glob(ARCHIVE_GLOB) if "-diagnostic" not in p.name and _archive_day(p))
    newest_per_day: dict[dt.date, Path] = {}
    for p in archives:
        newest_per_day[_archive_day(p)] = p
    keep = {newest_per_day[d] for d in sorted(newest_per_day)[-days:]}
    newest_per_week: dict[tuple, Path] = {}
    for d in sorted(newest_per_day):
        newest_per_week[d.isocalendar()[:2]] = newest_per_day[d]
    keep |= {newest_per_week[w] for w in sorted(newest_per_week)[-weeks:]}
    if archives:
        keep.add(archives[-1])
    removed = []
    for p in archives:
        if p not in keep:
            p.unlink()
            removed.append(p.name)
    return removed


# ---------------------------------------------------------------------------------------------- restore drill
def restore_drill(settings: OpsSettings | None = None, archive: Path | None = None, *,
                  start_server: bool = True, port: int = 8799) -> dict:
    """Decrypt into a throw-away directory, verify the exact inventory; with ``start_server`` also serve it on a
    temporary port and compare item counts (``level='service'``), else ``level='files'`` only. Never touches the
    live store; never recreates a missing key."""
    settings = settings or OpsSettings.from_env()
    t0 = time.time()
    record: dict = {"ok": False, "level": "service" if start_server else "files"}
    try:
        key = load_key(settings.backup_key)
        candidates = sorted(p for p in settings.backup_dir.glob(ARCHIVE_GLOB) if "-diagnostic" not in p.name)
        archive = archive or (candidates[-1] if candidates else None)
        if archive is None:
            raise BackupError("no qualified archive found")
        record["archive"] = archive.name
        with tempfile.TemporaryDirectory(prefix="mosr-") as tmp:
            root = extract(archive, key, Path(tmp))
            manifest, problems = verify_manifest(root)
            created = dt.datetime.strptime(manifest["created"], "%Y%m%d-%H%M%S")
            record.update(manifest_problems=problems, qualified=manifest.get("qualified", False),
                          backup_age_hours=round((dt.datetime.now() - created).total_seconds() / 3600, 2),
                          expected_live_total=manifest["items"]["live_total"])
            if start_server and not problems:
                record["server_check"] = _serve_and_count(root, settings, port, manifest["items"]["live_total"])
            record["ok"] = not problems and (not start_server or record.get("server_check", {}).get("match") is True)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["duration_s"] = round(time.time() - t0, 1)
    _log(settings, "restore_drill.jsonl", record)
    return record


def _serve_and_count(root: Path, settings: OpsSettings, port: int, expected: int) -> dict:
    import sys

    import anyio

    from .auth import call_tool

    env = dict(os.environ, MEMORY_V4_DIR=str(root), MEMORY_V4_PROJECT="*", MEMORY_V4_TOKEN_FILE=str(settings.token_file))
    proc = subprocess.Popen([sys.executable, "-m", "memory_os_v4.server", "--transport", "http", "--host",
                             "127.0.0.1", "--port", str(port)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_settings = OpsSettings(**{**settings.__dict__, "url": f"http://127.0.0.1:{port}/mcp/"})
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                return {"error": f"temporary server exited ({proc.returncode})"}
            try:
                status = anyio.run(call_tool, "memory_status", {}, tmp_settings)
                return {"status_total": status["total"], "expected_total": expected,
                        "match": status["total"] == expected}
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
