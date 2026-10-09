"""Final evaluation must reject leakage, changed artifacts and incomplete plans."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bench.evaluation_manifest import ManifestError, validate


def complete_manifest(tmp_path):
    def artifact(name, content):
        path = tmp_path / name
        path.write_text(content)
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    split = {"dev": [{"id": "dev-q1", "conversation_id": "c1", "user_id": "u1", "repository_id": "r1"}],
             "test": [{"id": "test-q1", "conversation_id": "c2", "user_id": "u2", "repository_id": "r2"}]}
    return {
        "protocol_version": "1.0", "stage": "final", "run_id": "synthetic-final",
        "source": {"commit": "a" * 40, "artifact": artifact("candidate.whl", "candidate")},
        "environment_lock": artifact("requirements.lock", "numpy==1.26.4"), "seed": 17,
        "hardware": {"os": "Ubuntu", "cpu": "fixture", "accelerator": "none", "device": "cpu", "python": "3.11.9", "ram_gib": 16},
        "timing": {"mode": "live", "cache": "both", "repetitions": 3, "warmup_queries": 2, "scope": "ingest+retrieval+generation"},
        "models": {role: {"enabled": False} for role in ("embedding", "reranker", "planner", "response", "judge")},
        "tasks": [{"kind": "retrieval", "metrics": ["recall_at_k", "mrr"], "scoring_rule": "binary evidence ids; macro by category"}],
        "top_k": 10,
        "budgets": {"memory_bytes": 1024, "ingest_tokens": 100, "context_tokens": 100, "query_tokens": 100, "model_calls": 1, "wall_seconds": 60, "api_usd": 0},
        "systems": [{"name": name, "revision": "a" * 40, "command": "python -m fixture", "config": {"mode": "bm25"}, "budget_policy": "shared"} for name in ("baseline", "candidate")],
        "datasets": [{"name": "synthetic", "revision": "v1", "subset": "all", "exclusions": "none", "data": artifact("data.json", "{}"), "splits": artifact("splits.json", json.dumps(split)), "group_keys": ["conversation_id", "user_id", "repository_id"], "test_count": 1}],
        "selection_rule": "dev_only_then_freeze", "reporting": "separate_task_families",
    }


def test_complete_manifest_is_accepted(tmp_path):
    validate(complete_manifest(tmp_path), tmp_path)


@pytest.mark.parametrize("key", ["models", "budgets", "hardware", "environment_lock", "tasks", "systems", "datasets"])
def test_missing_configuration_rejected(tmp_path, key):
    manifest = complete_manifest(tmp_path)
    del manifest[key]
    with pytest.raises(ManifestError):
        validate(manifest, tmp_path)


@pytest.mark.parametrize("group_key", ["conversation_id", "user_id", "repository_id"])
def test_group_leakage_rejected_even_when_question_ids_differ(tmp_path, group_key):
    manifest = complete_manifest(tmp_path)
    path = tmp_path / "splits.json"
    split = json.loads(path.read_text())
    split["test"][0][group_key] = split["dev"][0][group_key]
    path.write_text(json.dumps(split))
    manifest["datasets"][0]["splits"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ManifestError, match="leaks"):
        validate(manifest, tmp_path)


def test_changed_dataset_rejected(tmp_path):
    manifest = complete_manifest(tmp_path)
    (tmp_path / "data.json").write_text("changed")
    with pytest.raises(ManifestError, match="checksum"):
        validate(manifest, tmp_path)


def test_response_metric_cannot_be_presented_as_retrieval(tmp_path):
    manifest = complete_manifest(tmp_path)
    manifest["tasks"][0]["metrics"] = ["answer_accuracy"]
    with pytest.raises(ManifestError, match="task family"):
        validate(manifest, tmp_path)


def test_floating_model_revision_rejected(tmp_path):
    manifest = complete_manifest(tmp_path)
    manifest["models"]["embedding"] = {"enabled": True, "id": "real/model", "revision": "main", "provider": "local", "precision": "float32", "prompt_sha256": "a" * 64}
    with pytest.raises(ManifestError, match="floating"):
        validate(manifest, tmp_path)


def test_freeze_refuses_overwrite_and_check_rejects_retuned_config(tmp_path):
    manifest = complete_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    lock = tmp_path / "frozen.json"
    script = Path(__file__).resolve().parents[1] / "v4/bench/evaluation_manifest.py"
    command = [sys.executable, str(script), str(path)]
    assert subprocess.run(command + ["--freeze", str(lock)], capture_output=True).returncode == 0
    assert subprocess.run(command + ["--freeze", str(lock)], capture_output=True).returncode == 2
    assert subprocess.run(command + ["--check-lock", str(lock)], capture_output=True).returncode == 0
    changed = copy.deepcopy(manifest)
    changed["top_k"] = 20
    path.write_text(json.dumps(changed))
    assert subprocess.run(command + ["--check-lock", str(lock)], capture_output=True).returncode == 2
