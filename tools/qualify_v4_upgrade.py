"""Offline upgrade/backup/rollback exercise on synthetic data only.

Pass --baseline-pythonpath for an extracted published 4.0.0 wheel and
--candidate-pythonpath for the candidate package. Uses the current Python
environment's dependencies, not a clean installation qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(pythonpath: str, directory: Path, code: str) -> dict:
    env = {**os.environ, "PYTHONPATH": pythonpath, "MEMORY_V4_EMBED_MODEL": "hash",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code, str(directory)], env=env,
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def hashes(directory: Path) -> dict:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


SETUP = '''
import json, sys
from memory_os_v4 import MemoryV4, Config
m = MemoryV4(config=Config(data_dir=sys.argv[1], embed_model="hash"))
episode = m.add_episode("Synthetic source episode", project="alpha")
fact = m.add_facts([{"text": "Synthetic user prefers Python"}], project="alpha")["added"][0]
m.blocks.set("project_state", "Legacy unscoped state")
m.flush()
print(json.dumps({"episode_id": episode.id, "fact_id": fact, "count": m.status()["total"]}))
'''

UPGRADE = '''
import json, sys
from memory_os_v4 import MemoryV4, Config
with MemoryV4(config=Config(data_dir=sys.argv[1], embed_model="hash")) as m:
    before = m.status()["total"]
    assert before == 2
    assert m.blocks.get("project_state", project="default") == "Legacy unscoped state"
    assert not m.session_brief(project="beta")["blocks"].get("project_state")
    original_ids = sorted(m.store.items)
    identity = {"subject_id": "user:synthetic", "predicate": "home_city", "cardinality": "single"}
    old = m.add_facts([{"text": "Synthetic user lives in Paris", "identity": identity}],
                      project="alpha", event_time=100)["added"][0]
    new = m.add_facts([{"text": "Synthetic user lives in Lyon", "identity": identity}],
                      project="alpha", event_time=200)["added"][0]
    assert m.store.get(old).invalidated_by == new
    assert m.store.get(old).valid_until == 200
    count = m.status()["total"]
with MemoryV4(config=Config(data_dir=sys.argv[1], embed_model="hash")) as m:
    assert m.status()["total"] == count
    assert all(m.store.get(i) for i in original_ids)
    assert m.store.get(old).invalidated_by == new
    print(json.dumps({"source_count": before, "upgraded_count": count,
                      "original_ids": original_ids, "reopen_ok": True,
                      "legacy_blocks_confined_to_default": True}))
'''

RESTORE = '''
import json, sys
from memory_os_v4 import MemoryV4, Config
m = MemoryV4(config=Config(data_dir=sys.argv[1], embed_model="hash"))
assert m.status()["total"] == 2
assert m.blocks.get("project_state") == "Legacy unscoped state"
print(json.dumps({"ids": sorted(m.store.items), "count": m.status()["total"]}))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pythonpath", required=True)
    parser.add_argument("--candidate-pythonpath", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="memory-upgrade-") as temp:
        root = Path(temp)
        store = root / "store"
        baseline = run(args.baseline_pythonpath, store, SETUP)
        backup = root / "backup"
        # The baseline process exited: no active writer during the snapshot.
        shutil.copytree(store, backup)
        backup_hashes = hashes(backup)
        candidate = run(args.candidate_pythonpath, store, UPGRADE)
        assert hashes(backup) == backup_hashes
        restored = root / "restored"
        shutil.copytree(backup, restored)
        assert hashes(restored) == backup_hashes
        rollback = run(args.baseline_pythonpath, restored, RESTORE)
        assert rollback["ids"] == candidate["original_ids"]
        print(json.dumps({"baseline": baseline, "candidate": candidate,
                          "rollback": rollback, "backup_unchanged": True,
                          "restore_byte_identical_before_open": True,
                          "scope": "synthetic hash-model data; stopped-writer full-directory backup"}, indent=2))


if __name__ == "__main__":
    main()
