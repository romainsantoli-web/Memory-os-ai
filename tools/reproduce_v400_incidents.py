"""Reproduce the three audited 4.0.0 incidents against an explicit import path.

Run with PYTHONPATH pointing at an extracted 4.0.0 wheel. All stores are
temporary. Exit zero means all historical incidents were reproduced, not that
the engine is healthy. The corrected behavior belongs in the regression suite.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import memory_os_v4
from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4


def run() -> dict:
    observations = {}
    with tempfile.TemporaryDirectory(prefix="memory-v400-incidents-") as tmp:
        root = Path(tmp)
        memory = MemoryV4(config=Config(data_dir=str(root / "dedup"), embed_model="hash"))
        first = memory.add_facts([{"text": "Romain prefers Python."}], project="alpha")
        second = memory.add_facts([{"text": "Romain prefers Python."}], project="beta")
        observations["cross_project_exact_dedup"] = {
            "reproduced": not second["added"] and second["deduped"] == first["added"],
            "alpha_added": len(first["added"]),
            "beta_added": len(second["added"]),
            "beta_deduped_to_alpha": second["deduped"] == first["added"],
        }

        memory = MemoryV4(config=Config(data_dir=str(root / "consolidation"), embed_model="hash"))
        old = memory.add_facts([{
            "text": "Romain uses PostgreSQL for the primary production database deployed on the shared server in Paris.",
            "entities": ["Romain", "PostgreSQL"],
        }], project="alpha", event_time=100)["added"][0]
        new = memory.add_facts([{
            "text": "Romain uses SQLite for the primary production database deployed on the shared server in Paris.",
            "entities": ["Romain", "SQLite"],
        }], project="beta", event_time=200)["added"][0]
        before = memory.store.get(old).invalidated_by
        memory.consolidate()
        after = memory.store.get(old).invalidated_by
        observations["cross_project_consolidation"] = {
            "reproduced": before is None and after == new,
            "alpha_invalidated_by_beta": after == new,
        }

        memory = MemoryV4(config=Config(data_dir=str(root / "blocks"), embed_model="hash"))
        memory.blocks.set("project_state", "Alpha private project state")
        leaked = memory.session_brief(project="beta")["blocks"].get("project_state")
        observations["unscoped_block_in_project_brief"] = {
            "reproduced": leaked == "Alpha private project state",
            "beta_receives_unscoped_block": bool(leaked),
        }
    return {
        "imported_package": str(Path(memory_os_v4.__file__).resolve()),
        "purpose": "historical reproduction; true indicates an incident exists",
        "incidents": observations,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(x["reproduced"] for x in result["incidents"].values()) else 1)
