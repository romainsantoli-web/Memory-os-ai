"""Strict structural validation of a JSONL memory-os-v4 store, without loading an engine (A6 R4b).

Mirrors the engine's own rules (``Store.check_integrity``, ``readiness.engine_readiness``) so a backup is only
qualified when the copied store is one the engine would accept: JSON objects without duplicate keys or non-finite
numbers, newline-terminated logs, item ids/kinds/text/projects/temporal fields, block scopes/names/text, request
receipts. Valid *pending* receipts stay uncertain (counted, never turned into a success). Derived index files are
only required to be well-formed JSON when they are ``.json``. Unknown fields are preserved (never rewritten).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..store import VALID_KINDS, MemoryItem

TEMPORAL_FIELDS = ("created_at", "event_time", "valid_from", "valid_until", "last_accessed")


class StoreCheckError(ValueError):
    pass


def _no_dup(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise StoreCheckError("duplicate JSON key")
    return dict(pairs)


def _non_finite(token: str):
    raise StoreCheckError(f"non-finite number {token}")


def loads_strict(data: bytes | str):
    return json.loads(data, object_pairs_hook=_no_dup, parse_constant=_non_finite)


def _jsonl(name: str, data: bytes) -> list[dict]:
    if data and not data.endswith(b"\n"):
        raise StoreCheckError(f"{name}: truncated (no final newline) — uncertain result")
    rows = []
    for n, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = loads_strict(line)
        except (ValueError, StoreCheckError) as exc:
            raise StoreCheckError(f"{name} line {n}: invalid JSON ({exc})") from None
        if not isinstance(row, dict):
            raise StoreCheckError(f"{name} line {n}: not an object")
        rows.append(row)
    return rows


def _valid_project(value) -> bool:
    return isinstance(value, str) and bool(value) and value != "*"


def check_items(data: bytes) -> dict:
    last: dict[str, dict] = {}
    for n, row in enumerate(_jsonl("items.jsonl", data), 1):
        try:
            item = MemoryItem.from_dict(row)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise StoreCheckError(f"items.jsonl record {n}: {type(exc).__name__}") from None
        if not isinstance(item.id, str) or not item.id:
            raise StoreCheckError(f"items.jsonl record {n}: empty or non-string id")
        if item.kind not in VALID_KINDS:
            raise StoreCheckError(f"items.jsonl record {n}: unknown kind {item.kind!r}")
        if not isinstance(item.text, str):
            raise StoreCheckError(f"items.jsonl record {n}: text is not a string")
        if not _valid_project(item.project):
            raise StoreCheckError(f"items.jsonl record {n}: invalid project")
        for field in TEMPORAL_FIELDS:
            value = row.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise StoreCheckError(f"items.jsonl record {n}: invalid {field}")
        last[item.id] = row
    by_kind: dict[str, int] = {}
    for row in last.values():
        if not row.get("deleted"):
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    return {"live_by_kind": by_kind, "live_total": sum(by_kind.values()), "ids": len(last)}


def check_blocks(data: bytes) -> int:
    rows = _jsonl("blocks.jsonl", data)
    for n, row in enumerate(rows, 1):
        scope = row.get("scope", "project")
        if scope not in {"global", "project"} or not isinstance(row.get("name"), str) \
                or not isinstance(row.get("text"), str):
            raise StoreCheckError(f"blocks.jsonl line {n}: invalid scope/name/text")
        if scope == "project" and not _valid_project(row.get("project", "default")):
            raise StoreCheckError(f"blocks.jsonl line {n}: invalid project")
        updated = row.get("updated_at")
        if updated is not None and (not isinstance(updated, (int, float)) or not math.isfinite(updated)):
            raise StoreCheckError(f"blocks.jsonl line {n}: invalid updated_at")
    return len(rows)


def check_requests(data: bytes) -> dict:
    rows = _jsonl("requests.jsonl", data)
    pending = 0
    for n, row in enumerate(rows, 1):
        if not isinstance(row.get("scope"), str) or not isinstance(row.get("request_id"), str) \
                or not row["request_id"]:
            raise StoreCheckError(f"requests.jsonl line {n}: invalid scope/request_id")
        if row.get("status") not in {None, "complete", "completed", "done"}:
            pending += 1
    return {"receipts": len(rows), "uncertain_receipts": pending}


def check_files(files: dict[str, bytes]) -> dict:
    """Validate an in-memory snapshot {relative path: bytes}. Returns the counts recorded in the manifest."""
    if "items.jsonl" not in files:
        raise StoreCheckError("items.jsonl missing")
    report: dict = {"items": check_items(files["items.jsonl"])}
    if "blocks.jsonl" in files:
        report["blocks"] = check_blocks(files["blocks.jsonl"])
    if "requests.jsonl" in files:
        report["requests"] = check_requests(files["requests.jsonl"])
    if "meta.json" in files and not isinstance(loads_strict(files["meta.json"]), dict):
        raise StoreCheckError("meta.json is not an object")
    for rel, data in files.items():
        if rel.endswith(".json") and rel != "meta.json":
            try:
                loads_strict(data)
            except (ValueError, StoreCheckError) as exc:
                raise StoreCheckError(f"{rel}: invalid JSON ({exc})") from None
    return report


def check_dir(root: Path, names: list[str]) -> dict:
    return check_files({rel: (root / rel).read_bytes() for rel in names})
