"""Real default-model semantic retrieval, persisted and reloaded on CPU.

Uses cached weights in offline mode; CI downloads weights in a separate step.
MEMORY_V4_EMBED_MODEL may select another model for diagnostics. CI leaves it unset.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memory-real-model-") as directory:
        model_name = os.environ.get("MEMORY_V4_EMBED_MODEL", Config().embed_model)
        assert model_name != "hash", "default-model qualification requires real semantic embeddings"
        config = Config(
            data_dir=directory, embed_model=model_name,
            device=os.environ.get("MEMORY_V4_DEVICE", "cpu"),
            use_bm25=False, use_entities=False, use_recency=False,
        )
        memory = MemoryV4(config=config)
        memory.add_episode("The automobile broke down and a mechanic repaired its motor.", project="smoke")
        memory.add_episode("A gardener watered the flowers in the backyard.", project="smoke")
        memory.add_episode("The accountant submitted the quarterly tax return.", project="smoke")
        query = "Who fixed the engine of the car?"
        hits = memory.search(query, top_k=1)
        assert hits and "mechanic" in hits[0].item.text, hits
        assert memory.status()["dim"] > 64
        memory.flush()
        assert any(Path(directory).iterdir()), "flush must persist the memory"
        dimension = memory.status()["dim"]
        memory.close()
        reloaded = MemoryV4(config=config)
        hits = reloaded.search(query, top_k=1)
        assert hits and "mechanic" in hits[0].item.text, hits
        assert reloaded.status()["dim"] == dimension
        reloaded.close()
        print(f"Real semantic persistence/retrieval passed: {model_name}")


if __name__ == "__main__":
    main()
