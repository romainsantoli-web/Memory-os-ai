"""Daily health report (PLAN §10.1): read-only scan of the store log plus server, backup, disk and memory checks.

Supersessions are compared with a baseline (``known_supersessions.json`` in the ops directory): while automatic
supersession is confined (``MEMORY_V4_SUPERSEDE_COS`` > 1 on the server), any *new* one is an S1 anomaly.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import time
import urllib.request

from .auth import auth_headers
from .settings import OpsSettings

DISK_MIN_GIB = 5.0
BACKUP_MAX_AGE_H = 26.0


def scan_store(items_path) -> dict:
    last: dict[str, dict] = {}
    added_24h: dict[str, int] = {}
    forgotten_24h = malformed = 0
    day = time.time() - 86400
    with open(items_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            prev = last.get(d["id"])
            if prev is None and d.get("created_at", 0) >= day:
                added_24h[d.get("kind", "?")] = added_24h.get(d.get("kind", "?"), 0) + 1
            if prev is not None and d.get("deleted") and not prev.get("deleted"):
                forgotten_24h += 1
            last[d["id"]] = d
    live = [d for d in last.values() if not d.get("deleted")]
    return {
        "live_total": len(live),
        "facts_valid": sum(1 for d in live if d.get("kind") == "fact" and not d.get("valid_until")),
        "added_24h": added_24h,
        "forgotten_24h": forgotten_24h,
        "malformed_lines": malformed,
        "superseded_ids": sorted(d["id"] for d in last.values() if d.get("invalidated_by")),
    }


def report(settings: OpsSettings | None = None, confined: bool | None = None) -> dict:
    settings = settings or OpsSettings.from_env()
    if confined is None:
        confined = float(os.environ.get("MEMORY_V4_SUPERSEDE_COS", "0.8")) > 1.0
    anomalies: list[str] = []
    r: dict = {"at": dt.datetime.now().isoformat(timespec="seconds")}
    try:
        req = urllib.request.Request(settings.health_url, headers=auth_headers(settings))
        with urllib.request.urlopen(req, timeout=3) as resp:
            r["health"] = json.loads(resp.read())
    except Exception as exc:
        r["health"] = {"ok": False, "error": type(exc).__name__}
        anomalies.append("server unreachable")
    scan = scan_store(settings.store_dir / "items.jsonl")
    sup = scan.pop("superseded_ids")
    r.update(scan)
    if scan["malformed_lines"]:
        anomalies.append(f"{scan['malformed_lines']} malformed line(s) in items.jsonl")
    settings.ops_dir.mkdir(parents=True, exist_ok=True)
    known_file = settings.ops_dir / "known_supersessions.json"
    if not known_file.exists():
        known_file.write_text(json.dumps({"baseline_at": r["at"], "ids": sup}, indent=1))
    known = set(json.loads(known_file.read_text())["ids"])
    r["superseded_total"] = len(sup)
    r["superseded_new"] = [i for i in sup if i not in known]
    if confined and r["superseded_new"]:
        anomalies.append(f"{len(r['superseded_new'])} new supersession(s) while confined (S1)")
    age = None
    log = settings.ops_dir / "logs/backup.jsonl"
    if log.exists():
        ok = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        ok = [b for b in ok if b.get("ok")]
        if ok:
            age = round(
                (dt.datetime.now() - dt.datetime.strptime(ok[-1]["stamp"], "%Y%m%d-%H%M%S")).total_seconds()
                / 3600,
                1,
            )
    r["last_backup_age_h"] = age
    if age is None or age > BACKUP_MAX_AGE_H:
        anomalies.append(f"backup missing or stale (age {age} h)")
    r["disk_free_gib"] = round(shutil.disk_usage(settings.store_dir).free / 2**30, 1)
    if r["disk_free_gib"] < DISK_MIN_GIB:
        anomalies.append(f"free disk {r['disk_free_gib']} GiB < {DISK_MIN_GIB}")
    r["anomalies"] = anomalies
    reports = settings.ops_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "health.jsonl").open("a") as fh:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (reports / f"{dt.date.today().isoformat()}.md").write_text(render(r))
    return r


def render(r: dict) -> str:
    lines = [
        f"# memory-os-v4 health — {r['at']}",
        "",
        f"- Server: {'OK' if r['health'].get('ok') else 'DOWN'} · live items {r['live_total']} · "
        f"valid facts {r['facts_valid']}",
        f"- Added 24 h: {r['added_24h']} · forgotten 24 h: {r['forgotten_24h']} · supersessions "
        f"{r['superseded_total']} (new {len(r['superseded_new'])})",
        f"- Backup age: {r['last_backup_age_h']} h · free disk {r['disk_free_gib']} GiB",
        "",
        "## Anomalies",
        *(f"- {a}" for a in r["anomalies"] or ["none"]),
    ]
    return "\n".join(lines) + "\n"


def notify(r: dict) -> None:
    if not r["anomalies"] or shutil.which("osascript") is None:
        return
    msg = "; ".join(r["anomalies"])[:200].replace('"', "'")
    subprocess.run(
        ["osascript", "-e", f'display notification "{msg}" with title "memory-os anomaly"'],
        capture_output=True,
        check=False,
    )
