"""Strict, dependency-free preflight for frozen final evaluations (protocol v1)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


class ManifestError(ValueError):
    """The planned final evaluation is incomplete or cannot be reproduced."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def text(value: object, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label}: nonempty string required")
    require(value.lower() not in {"tbd", "todo", "unknown", "replace-me"}, f"{label}: placeholder forbidden")
    return value


def mapping(value: object, label: str) -> dict:
    require(isinstance(value, dict), f"{label}: object required")
    return value


def entries(value: object, label: str) -> list:
    require(isinstance(value, list) and bool(value), f"{label}: nonempty array required")
    return value


def positive(value: object, label: str, zero: bool = False) -> None:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: number required")
    require(value >= 0 if zero else value > 0, f"{label}: invalid budget/value")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_record(record: object, base: Path, label: str) -> Path:
    record = mapping(record, label)
    path = base / text(record.get("path"), f"{label}.path")
    expected = text(record.get("sha256"), f"{label}.sha256")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected)), f"{label}: SHA-256 required")
    require(path.is_file(), f"{label}: missing file {path}")
    require(digest(path) == expected, f"{label}: file checksum mismatch")
    return path


def validate(manifest: object, base: Path) -> None:
    manifest = mapping(manifest, "manifest")
    require(manifest.get("protocol_version") == "1.0", "protocol_version must be 1.0")
    require(manifest.get("stage") == "final", "final preflight requires stage=final")
    text(manifest.get("run_id"), "run_id")
    source = mapping(manifest.get("source"), "source")
    revision = text(source.get("commit"), "source.commit")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "source.commit must be an exact Git commit")
    file_record(source.get("artifact"), base, "source.artifact")
    file_record(manifest.get("environment_lock"), base, "environment_lock")
    require(
        isinstance(manifest.get("seed"), int) and not isinstance(manifest["seed"], bool),
        "integer seed required",
    )
    hardware = mapping(manifest.get("hardware"), "hardware")
    for key in ("os", "cpu", "accelerator", "device", "python"):
        text(hardware.get(key), f"hardware.{key}")
    positive(hardware.get("ram_gib"), "hardware.ram_gib")
    timing = mapping(manifest.get("timing"), "timing")
    require(timing.get("mode") in {"live", "offline_replay"}, "timing.mode must be live or offline_replay")
    require(timing.get("cache") in {"cold", "warm", "both"}, "timing.cache required")
    positive(timing.get("repetitions"), "timing.repetitions")
    positive(timing.get("warmup_queries"), "timing.warmup_queries", zero=True)
    require(timing.get("scope") == "ingest+retrieval+generation", "timing.scope must include all stages")
    models = mapping(manifest.get("models"), "models")
    for role in ("embedding", "reranker", "planner", "response", "judge"):
        model = mapping(models.get(role), f"models.{role}")
        require(isinstance(model.get("enabled"), bool), f"models.{role}.enabled required")
        if model["enabled"]:
            for key in ("id", "revision", "provider", "precision", "prompt_sha256"):
                text(model.get(key), f"models.{role}.{key}")
            require(
                bool(re.fullmatch(r"[0-9a-f]{64}", model["prompt_sha256"])),
                f"models.{role}: prompt hash required",
            )
            require(
                model["revision"] not in {"main", "latest", "master"},
                f"models.{role}: floating revision forbidden",
            )
    tasks = entries(manifest.get("tasks"), "tasks")
    kinds = set()
    metric_families = {
        "retrieval": {"recall_at_k", "mrr", "ndcg_at_k"},
        "response": {"answer_accuracy", "groundedness", "abstention_accuracy"},
        "agent_task": {"task_success", "constraint_violations", "tool_calls"},
    }
    for task in tasks:
        task = mapping(task, "task")
        kind = task.get("kind")
        require(kind in metric_families, "task.kind must be retrieval, response or agent_task")
        require(kind not in kinds, "duplicate task kind")
        kinds.add(kind)
        metrics = entries(task.get("metrics"), "task.metrics")
        require(all(isinstance(metric, str) for metric in metrics), "metric names must be strings")
        require(set(metrics) <= metric_families[kind], f"{kind}: metrics belong to a different task family")
        text(task.get("scoring_rule"), "task.scoring_rule")
        if kind != "retrieval":
            require(models["response"]["enabled"], "response and agent tasks require a response model")
    if "retrieval" in kinds:
        positive(manifest.get("top_k"), "top_k")
    budgets = mapping(manifest.get("budgets"), "budgets")
    for key in (
        "memory_bytes",
        "ingest_tokens",
        "context_tokens",
        "query_tokens",
        "model_calls",
        "wall_seconds",
    ):
        positive(budgets.get(key), f"budgets.{key}")
    positive(budgets.get("api_usd"), "budgets.api_usd", zero=True)
    systems = entries(manifest.get("systems"), "systems")
    require(len(systems) >= 2, "comparison needs at least two systems")
    names = set()
    for system in systems:
        system = mapping(system, "system")
        name = text(system.get("name"), "system.name")
        require(name not in names, "duplicate system name")
        names.add(name)
        text(system.get("revision"), f"{name}.revision")
        text(system.get("command"), f"{name}.command")
        require(
            bool(mapping(system.get("config"), f"{name}.config")),
            f"{name}: complete resolved config required",
        )
        require(system.get("budget_policy") == "shared", f"{name}: shared budgets required")
    dataset_names = set()
    for dataset in entries(manifest.get("datasets"), "datasets"):
        dataset = mapping(dataset, "dataset")
        name = text(dataset.get("name"), "dataset.name")
        require(name not in dataset_names, "duplicate dataset name")
        dataset_names.add(name)
        text(dataset.get("revision"), f"{name}.revision")
        text(dataset.get("subset"), f"{name}.subset")
        text(dataset.get("exclusions"), f"{name}.exclusions (use 'none' if empty)")
        file_record(dataset.get("data"), base, f"{name}.data")
        path = file_record(dataset.get("splits"), base, f"{name}.splits")
        try:
            split = mapping(json.loads(path.read_text()), f"{name}.splits")
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(f"{name}: invalid split file: {error}") from error
        group_keys = entries(dataset.get("group_keys"), f"{name}.group_keys")
        require(
            set(group_keys) <= {"conversation_id", "user_id", "repository_id"},
            f"{name}: invalid grouping key",
        )
        seen_rows = set()
        groups = {key: {} for key in group_keys}
        for partition in ("dev", "test"):
            for row in entries(split.get(partition), f"{name}.{partition}"):
                row = mapping(row, f"{name}.{partition}.row")
                row_id = text(row.get("id"), f"{name}.row.id")
                require(row_id not in seen_rows, f"{name}: duplicate/overlapping sample {row_id}")
                seen_rows.add(row_id)
                for key in group_keys:
                    group = text(row.get(key), f"{name}.{row_id}.{key}")
                    previous = groups[key].setdefault(group, partition)
                    require(previous == partition, f"{name}: {key}={group} leaks between dev and test")
        require(dataset.get("test_count") == len(split["test"]), f"{name}: test_count mismatch")
    require(
        manifest.get("selection_rule") == "dev_only_then_freeze", "selection_rule must freeze before test"
    )
    require(
        manifest.get("reporting") == "separate_task_families",
        "retrieval, response and task scores must remain separate",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", type=Path, help="create a new immutable pre-test lock (refuses overwrite)")
    mode.add_argument("--check-lock", type=Path, help="validate configuration/files against existing lock")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        validate(manifest, args.manifest.resolve().parent)
        fingerprint = digest(args.manifest)
        if args.freeze:
            lock = {
                "protocol_version": "1.0",
                "manifest_sha256": fingerprint,
                "frozen_at": datetime.now(timezone.utc).isoformat(),
            }
            with args.freeze.open("x") as handle:
                json.dump(lock, handle, indent=2)
                handle.write("\n")
        else:
            lock = mapping(json.loads(args.check_lock.read_text()), "lock")
            require(lock.get("protocol_version") == "1.0", "lock protocol mismatch")
            require(lock.get("manifest_sha256") == fingerprint, "manifest changed since freeze")
            text(lock.get("frozen_at"), "lock.frozen_at")
        print(f"Final evaluation preflight passed: {fingerprint}")
        return 0
    except (ManifestError, OSError, json.JSONDecodeError, TypeError) as error:
        parser.exit(2, f"Final evaluation rejected: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
