"""One end-to-end integration test using the real cached MiniLM embedder.

Marked slow (loads sentence-transformers + runs MPS/CPU inference) but still
executed by default, as required.
"""

import os
import time

import pytest

from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.mark.slow
def test_real_embedder_semantic_search_and_supersession(tmp_path):
    cfg = Config(
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        data_dir=str(tmp_path),
    )
    m = MemoryV4(config=cfg)

    m.add_episode("We picked FAISS FlatL2 for the vector index because it is exact.", project="p1")
    m.add_episode("The garden needs watering this weekend.", project="p1")
    m.add_episode("mm_cross_v2 fixed the fee accounting bug in the router.", project="p1")

    hits = m.search("Why did we choose FAISS for similarity search?", top_k=3)
    assert hits
    assert "FAISS" in hits[0].item.text

    # Real semantic embeddings should recognize the paraphrase without shared
    # tokens well enough to link the two location facts via supersession.
    t0 = time.time() - 200000
    t1 = time.time() - 100000
    m.add_facts([{"text": "Romain lives in Paris."}], project="p1", event_time=t0)
    result = m.add_facts([{"text": "Romain now lives in Lyon."}], project="p1", event_time=t1)
    assert result["superseded"], "semantic supersession should link the two location facts"

    m.flush()
    dim = m.status()["dim"]
    m.close()
    reloaded = MemoryV4(config=cfg)
    assert reloaded.status()["dim"] == dim == 384
    hits2 = reloaded.search("Where does Romain live now?", top_k=3)
    assert hits2
    assert "Lyon" in hits2[0].item.text
