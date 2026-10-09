"""Offline, append-only repair of INC-001. Simulation is the default.

Never loads an embedding model or rewrites historical records. Application
requires --offline, an existing writer.lock, and an exclusive maintenance lease.
The live store and original repository are deliberately forbidden even in dry run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

TARGETS = (
    "0513280149a6", "b474b5cfca5a", "d82927260504", "b6034e5343154",
    "2c2546b5c8e9", "14ef45923aef", "539f1faa61c9", "7d59e8ab4711", "3508868143ed",
)
CULPRIT = "e116128b7985433bb1605ea7cab5605b"
PROJECT = "Memory-os-ai"
TRACE = "repair-INC-001.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def guarded_directory(directory: Path) -> Path:
    directory = directory.resolve(strict=True)
    for forbidden in (Path.home() / ".memory-os-v4", Path.home() / "Memory-os-ai"):
        root = forbidden.resolve()
        if directory == root or root in directory.parents:
            raise ValueError("Production store and original repository are forbidden")
    for name in ("items.jsonl", "writer.lock", TRACE):
        path = directory / name
        if path.is_symlink() or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError("Symlinks and hard links are forbidden for repair files")
    return directory


def plan(data: bytes) -> tuple[list[dict], list[dict]]:
    if not data.endswith(b"\n"):
        raise ValueError("Log must end with a newline; truncated logs require recovery")
    history: dict[str, list[dict]] = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        history.setdefault(record["id"], []).append(record)
    culprit = history.get(CULPRIT, [None])[-1]
    if not culprit or culprit.get("kind") != "fact" or culprit.get("project") != PROJECT:
        raise ValueError("Incident source is absent or unexpected")
    repaired, changes = [], []
    for prefix in TARGETS:
        ids = [key for key in history if key.startswith(prefix)]
        if len(ids) != 1:
            raise ValueError(f"Missing or ambiguous target {prefix}")
        versions = history[ids[0]]
        current = versions[-1]
        if (current.get("kind") != "fact" or current.get("project") != PROJECT
                or current.get("deleted") or current.get("invalidated_by") != CULPRIT
                or current.get("valid_until") != culprit.get("event_time")):
            raise ValueError(f"Target {prefix} no longer matches the incident")
        transitions = [i for i in range(1, len(versions))
                       if versions[i].get("invalidated_by") == CULPRIT
                       and versions[i - 1].get("invalidated_by") != CULPRIT]
        if len(transitions) != 1:
            raise ValueError(f"No unique pre-incident version for {prefix}")
        previous = versions[transitions[0] - 1]
        if previous.get("valid_until") is not None or previous.get("invalidated_by") is not None:
            raise ValueError(f"Target {prefix} was already invalid before the incident")
        restored = copy.deepcopy(current)
        restored["valid_until"] = previous.get("valid_until")
        restored["invalidated_by"] = previous.get("invalidated_by")
        repaired.append(restored)
        changes.append({"id": current["id"], "fields": {
            key: {"before": current.get(key), "after": restored.get(key)}
            for key in ("valid_until", "invalidated_by")}})
    return repaired, changes


def sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_trace(directory: Path, trace: dict, *, create: bool = False) -> None:
    path = directory / TRACE
    payload = (json.dumps(trace, indent=2, sort_keys=True) + "\n").encode()
    if create:
        with path.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    else:
        temporary = directory / (TRACE + ".tmp")
        with temporary.open("xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    sync_directory(directory)


def repair(directory: Path, *, apply: bool = False, offline: bool = False) -> dict:
    directory = guarded_directory(directory)
    path = directory / "items.jsonl"
    if not apply:
        data = path.read_bytes()
        _, changes = plan(data)
        return {"incident": "INC-001", "mode": "simulation", "count": len(changes),
                "input_sha256": digest(data), "changes": changes}
    if not offline:
        raise ValueError("Application requires explicit --offline maintenance acknowledgement")
    import fcntl  # This maintenance tool targets the qualified macOS/POSIX profile.

    # The same inode/lock as the candidate engine; never create/delete its lease.
    with (directory / "writer.lock").open("r+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (directory / TRACE).exists():
                raise ValueError("Repair trace already exists; reconcile it, never blindly replay")
            data = path.read_bytes()
            restored, changes = plan(data)
            appended = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode()
                                for row in restored)
            trace = {"incident": "INC-001", "status": "pending", "created_at": time.time(),
                     "cause": "entity/cosine supersession without matching fact identity",
                     "count": len(restored), "changes": changes,
                     "before_sha256": digest(data), "after_sha256": digest(data + appended),
                     "original_bytes": len(data), "appended_sha256": digest(appended)}
            write_trace(directory, trace, create=True)
            with path.open("ab") as fh:
                fh.write(appended)
                fh.flush()
                os.fsync(fh.fileno())
            if digest(path.read_bytes()) != trace["after_sha256"]:
                raise ValueError("Log changed unexpectedly; pending trace requires reconciliation")
            trace["status"] = "complete"
            write_trace(directory, trace)
            return {"incident": "INC-001", "mode": "applied", "count": len(restored),
                    "trace": TRACE, "after_sha256": trace["after_sha256"]}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        result = repair(args.data_dir, apply=args.apply, offline=args.offline)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Repair refused: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
