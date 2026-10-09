"""Read-only readiness: strict source checks, loaded model and index coverage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess


def build_identity() -> dict:
    from . import __version__

    result = {"version": __version__, "commit": None, "commit_source": "unknown", "source_dirty": None}
    configured = os.environ.get("MEMORY_V4_BUILD_COMMIT")
    if configured:
        if re.fullmatch(r"[0-9a-f]{40}", configured) is None:
            raise ValueError("MEMORY_V4_BUILD_COMMIT must be a complete lowercase Git SHA")
        return {**result, "commit": configured, "commit_source": "deployment_manifest"}
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists() or not (root / "v4/src/memory_os_v4").is_dir():
        return result
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
        dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--",
                                "v4/src", "src", "pyproject.toml"], timeout=2,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if re.fullmatch(r"[0-9a-f]{40}", commit) and dirty in {0, 1}:
            return {**result, "commit": commit, "commit_source": "checkout", "source_dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _jsonl(path: Path) -> list[dict]:
    data = path.read_bytes() if path.exists() else b""
    if data and not data.endswith(b"\n"):
        raise ValueError("Truncated source")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def engine_readiness(engine) -> tuple[bool, str]:
    if engine is None:
        return False, "engine_not_loaded"
    if not engine._lock.acquire(blocking=False):
        return False, "engine_busy"
    try:
        engine._ensure_open()
        if engine._writer_lock._file is None:
            return False, "writer_lease_missing"
        if engine.config.embed_model != "hash" and getattr(engine.embedder, "_model", None) is None:
            return False, "model_not_loaded"
        if not engine.store.check_integrity():
            return False, "store_integrity"
        blocks = {}
        for record in _jsonl(engine.blocks.path):
            scope = record.get("scope", "project")
            project = None if scope == "global" else record.get("project", "default")
            if (scope not in {"global", "project"} or not isinstance(record["name"], str)
                    or not isinstance(record["text"], str)
                    or (scope == "project" and (not isinstance(project, str) or not project or project == "*"))):
                return False, "blocks_integrity"
            blocks[(project, record["name"])] = record
        if set(blocks) != set(engine.blocks._blocks):
            return False, "blocks_integrity"
        for key, record in blocks.items():
            block = engine.blocks._blocks[key]
            if block.text != record["text"] or block.updated_at != record["updated_at"]:
                return False, "blocks_integrity"
        receipts = {}
        for record in _jsonl(engine.requests.path):
            receipts[(record["scope"], record["request_id"])] = record
        if engine.requests._poisoned or receipts != engine.requests.records:
            return False, "requests_integrity"
        live_ids = {item.id for item in engine.store.all(False)}
        indexed = {engine.store.chunk_owner.get(key, key) for key in engine.vector_index.ids()}
        if (indexed != live_ids or set(engine.bm25.doc_len) != live_ids
                or set(engine.entities.doc_entities) != live_ids or engine.facet_index.ids() != live_ids):
            return False, "indexes_not_ready"
        return True, "ready"
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError):
        return False, "integrity_check_failed"
    finally:
        engine._lock.release()
