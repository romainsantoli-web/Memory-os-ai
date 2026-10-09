"""Read the v3 conversation log (``_conversation_log.jsonl``) into v4 episode
and summary specs. Idempotent: deterministic ids derived from the record
content mean re-running a migration adds nothing already present.

Record format (v3): ``{timestamp, project, summary, message_count,
messages: [{role, content}]}``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("memory_os_v4")


def resolve_under_home(path_str: str) -> Path:
    """Resolve ``path_str`` and require it to live under $HOME or $MEMORY_WORKSPACE.

    A6 (adversarial review, 2026-09-12): ``server.py`` already guards
    ``memory_migrate_v3``'s ``log_path`` this way, but the CORE
    ``MemoryV4.migrate_v3`` / :func:`read_v3_log` had no guard of its own --
    calling the façade directly (any embedder, any script, not just the MCP
    server) let ``log_path`` read an arbitrary file (e.g. ``/etc/hosts``)
    outside the user's data. Kept as a shared helper so both layers guard
    identically instead of drifting; server.py imports this under its
    existing private name.
    """
    home = Path.home().resolve()
    roots = [home]
    workspace = os.environ.get("MEMORY_WORKSPACE")
    if workspace:
        roots.append(Path(workspace).expanduser().resolve())
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = home / p
    resolved = p.resolve()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"path traversal blocked: {path_str!r} does not resolve under $HOME or MEMORY_WORKSPACE")
    return resolved


def _parse_v3_timestamp(ts: str) -> float:
    """Parse a v3 ISO timestamp. Naive (no tzinfo) timestamps are treated as
    UTC (A6, adversarial review 2026-09-12) -- ``datetime.timestamp()`` on a
    naive datetime otherwise silently assumes the LOCAL system timezone,
    which would shift every migrated episode's event_time by the host's UTC
    offset."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _deterministic_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def read_v3_log(log_path: str, existing_items: dict) -> tuple[list[dict], list[dict], int]:
    """Parse a v3 JSONL log into (episode_specs, summary_specs, n_skipped).

    ``existing_items`` is the current store's id -> item map, used to skip
    records already migrated (idempotence). ``log_path`` must resolve under
    $HOME or $MEMORY_WORKSPACE (A6); raises ``ValueError`` otherwise.
    """
    path = resolve_under_home(log_path)
    episode_specs: list[dict] = []
    summary_specs: list[dict] = []
    skipped = 0

    if not path.exists():
        logger.warning("v3 log not found: %s", path)
        return episode_specs, summary_specs, skipped

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed v3 log line")
                continue

            timestamp = rec.get("timestamp")
            try:
                event_time = _parse_v3_timestamp(timestamp) if timestamp else None
            except ValueError:
                event_time = None
            project = rec.get("project", "default")

            summary = rec.get("summary")
            if summary:
                sid = _deterministic_id("v3sum", str(timestamp), project, summary)
                if sid in existing_items:
                    skipped += 1
                else:
                    summary_specs.append(
                        {
                            "item_id": sid,
                            "text": summary,
                            "kind": "summary",
                            "project": project,
                            "event_time": event_time,
                            "importance": 0.8,
                            "tags": ["v3", "summary"],
                        }
                    )

            for idx, msg in enumerate(rec.get("messages", [])):
                content = msg.get("content", "")
                role = msg.get("role", "user")
                eid = _deterministic_id("v3ep", str(timestamp), str(idx), role, content)
                if eid in existing_items:
                    skipped += 1
                    continue
                episode_specs.append(
                    {
                        "item_id": eid,
                        "text": content,
                        "role": role,
                        "project": project,
                        "event_time": event_time,
                        "source": "migrate_v3",
                    }
                )

    return episode_specs, summary_specs, skipped
