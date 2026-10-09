"""Offline, append-only targeted repair from a versioned plan. Dry run by default.

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
import re
import time

TARGETS = (
    "0513280149a6", "b474b5cfca5a", "d82927260504", "b6034e5343154",
    "2c2546b5c8e9", "14ef45923aef", "539f1faa61c9", "7d59e8ab4711", "3508868143ed",
)
CULPRIT = "e116128b7985433bb1605ea7cab5605b"
PROJECT = "Memory-os-ai"
TRACE = "repair-targeted.json"
DEFAULT_PLAN = Path(__file__).parent / "repair_plans" / "INC-001-001b-v1.json"


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


def load_plan(path: Path = DEFAULT_PLAN) -> dict:
    data = path.read_bytes()
    value = json.loads(data)
    if (value.get("schema_version") != 1 or not isinstance(value.get("plan_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_.+-]{1,80}", value["plan_id"])
            or not isinstance(value.get("targets"), list) or not 1 <= len(value["targets"]) <= 1000):
        raise ValueError("Invalid repair plan schema")
    seen = set()
    for target in value["targets"]:
        if not isinstance(target, dict):
            raise ValueError("Invalid repair target")
        for key in ("target_id", "invalidated_by"):
            if not isinstance(target.get(key), str) or not re.fullmatch(r"[a-f0-9]{32}", target[key]):
                raise ValueError("Repair plans require full 32-character IDs")
        if (target["target_id"] in seen or target["target_id"] == target["invalidated_by"]
                or not isinstance(target.get("project"), str) or not target["project"]
                or target["project"] == "*" or not isinstance(target.get("incident"), str)
                or not target["incident"]):
            raise ValueError("Duplicate or invalid repair target/project/incident")
        seen.add(target["target_id"])
    return {**value, "plan_sha256": digest(data)}


def plan(data: bytes, repair_plan: dict | None = None) -> tuple[list[dict], list[dict]]:
    repair_plan = repair_plan or load_plan()
    if not data.endswith(b"\n"):
        raise ValueError("Log must end with a newline; truncated logs require recovery")
    history: dict[str, list[dict]] = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        history.setdefault(record["id"], []).append(record)
    repaired, changes = [], []
    for target in repair_plan["targets"]:
        item_id, culprit_id, project = (target["target_id"], target["invalidated_by"], target["project"])
        culprit = history.get(culprit_id, [None])[-1]
        if not culprit or culprit.get("kind") != "fact" or culprit.get("project") != project:
            raise ValueError("Incident source is absent or unexpected")
        if item_id not in history:
            raise ValueError(f"Missing target {item_id}")
        versions = history[item_id]
        current = versions[-1]
        if (current.get("kind") != "fact" or current.get("project") != project
                or current.get("deleted") or current.get("invalidated_by") != culprit_id
                or current.get("valid_until") != culprit.get("event_time")):
            raise ValueError(f"Target {item_id} no longer matches the incident")
        transitions = [i for i in range(1, len(versions))
                       if versions[i].get("invalidated_by") == culprit_id
                       and versions[i - 1].get("invalidated_by") != culprit_id]
        if len(transitions) != 1:
            raise ValueError(f"No unique pre-incident version for {item_id}")
        previous = versions[transitions[0] - 1]
        if previous.get("valid_until") is not None or previous.get("invalidated_by") is not None:
            raise ValueError(f"Target {item_id} was already invalid before the incident")
        restored = copy.deepcopy(current)
        restored["valid_until"] = previous.get("valid_until")
        restored["invalidated_by"] = previous.get("invalidated_by")
        repaired.append(restored)
        changes.append({"id": current["id"], "incident": target["incident"], "project": project, "fields": {
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


def fingerprints(directory: Path) -> dict[str, str]:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Linked paths are forbidden in verification snapshots")
    return {str(p.relative_to(directory)): digest(p.read_bytes()) for p in files}


def verify(directory: Path, *, plan_path: Path = DEFAULT_PLAN, baseline: Path | None = None) -> dict:
    directory = guarded_directory(directory)
    repair_plan = load_plan(plan_path)
    trace = json.loads((directory / TRACE).read_bytes())
    if (trace.get("status") != "complete" or trace.get("plan_sha256") != repair_plan["plan_sha256"]
            or trace.get("plan_id") != repair_plan["plan_id"]):
        raise ValueError("Incomplete repair or mismatched plan")
    data = (directory / "items.jsonl").read_bytes()
    original = data[:trace["original_bytes"]]
    if digest(original) != trace["before_sha256"] or digest(data) != trace["after_sha256"]:
        raise ValueError("History or repaired log differs from the trace")
    restored, changes = plan(original, repair_plan)
    expected = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in restored)
    if data[len(original):] != expected or trace["changes"] != changes:
        raise ValueError("Repair differs from the validated target changes")
    if baseline is not None:
        before = json.loads(baseline.read_bytes())
        after = fingerprints(directory)
        if (set(after) != set(before) | {TRACE}
                or any(after.get(name) != value for name, value in before.items() if name != "items.jsonl")
                or before.get("items.jsonl") != digest(original)):
            raise ValueError("Unexpected changes outside targeted append and repair trace")
    return {"mode": "verified", "plan_id": repair_plan["plan_id"], "valid_facts": len(restored),
            "history_byte_prefix_preserved": True, "provenance_preserved": True,
            "other_items_unchanged": True, "other_files_checked": baseline is not None}


def repair(directory: Path, *, apply: bool = False, offline: bool = False,
           plan_path: Path = DEFAULT_PLAN) -> dict:
    directory = guarded_directory(directory)
    repair_plan = load_plan(plan_path)
    path = directory / "items.jsonl"
    if not apply:
        data = path.read_bytes()
        _, changes = plan(data, repair_plan)
        return {"plan_id": repair_plan["plan_id"], "mode": "simulation", "count": len(changes),
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
            restored, changes = plan(data, repair_plan)
            appended = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode()
                                for row in restored)
            trace = {"plan_id": repair_plan["plan_id"], "plan_sha256": repair_plan["plan_sha256"],
                     "status": "pending", "created_at": time.time(),
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
            return {"plan_id": repair_plan["plan_id"], "mode": "applied", "count": len(restored),
                    "trace": TRACE, "after_sha256": trace["after_sha256"]}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--verify", action="store_true", help="Read-only post-repair proof")
    parser.add_argument("--baseline-manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.verify and args.apply:
            raise ValueError("--verify and --apply are mutually exclusive")
        result = (verify(args.data_dir, plan_path=args.plan, baseline=args.baseline_manifest)
                  if args.verify else repair(args.data_dir, apply=args.apply,
                                             offline=args.offline, plan_path=args.plan))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Repair refused: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
