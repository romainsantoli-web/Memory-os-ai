"""Embedder / HashEmbedder unit tests (network-free: a fake SentenceTransformer
stands in for the real model -- see docs/ABLATIONS.md Palier 2-3 E10,
`Config.embed_max_seq_length`)."""

from __future__ import annotations

import numpy as np

from memory_os_v4.embed import Embedder, HashEmbedder, _resolve_device


class _FakeSentenceTransformer:
    """Records constructor args and `max_seq_length` writes; ``encode``
    returns deterministic unit vectors so ``Embedder`` never touches a real
    model in this test file."""

    instances: list["_FakeSentenceTransformer"] = []

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = -1  # sentinel: overwritten by Embedder if wired
        _FakeSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_embedder_stores_max_seq_length_before_model_load():
    e = Embedder("some/model", device="cpu", max_seq_length=256)
    assert e.max_seq_length == 256  # set in __init__, independent of lazy load


def test_embedder_defaults_max_seq_length_to_512():
    e = Embedder("some/model", device="cpu")
    assert e.max_seq_length == 512


def test_embedder_defaults_batch_size_to_32():
    e = Embedder("some/model", device="cpu")
    assert e.batch_size == 32


def test_embedder_passes_batch_size_to_model_encode(monkeypatch):
    import sentence_transformers

    from memory_os_v4 import embed as embed_mod

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(embed_mod, "_MODEL_CACHE", {})
    calls = []
    e = Embedder("some/model", device="cpu", batch_size=8)
    e._ensure_model()
    orig_encode = e._model.encode

    def spy_encode(texts, **kwargs):
        calls.append(kwargs.get("batch_size"))
        return orig_encode(texts, **kwargs)

    e._model.encode = spy_encode
    e.encode(["a", "b"])
    assert calls == [8]


def test_embedder_applies_max_seq_length_cap_to_loaded_model(monkeypatch):
    """E10: ibm-granite/granite-embedding-97m-multilingual-r2 reports a
    native max_seq_length of 32768 (ModernBERT-based) -- Embedder must cap
    it to Config.embed_max_seq_length right after loading, before any real
    encode() call."""
    import sentence_transformers

    from memory_os_v4 import embed as embed_mod

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(embed_mod, "_MODEL_CACHE", {})  # isolate from the process-wide model cache

    e = Embedder("ibm-granite/granite-embedding-97m-multilingual-r2", device="cpu", max_seq_length=512)
    e._ensure_model()
    assert e._model.max_seq_length == 512
    assert e.dim == 4


def test_embedder_reuses_cached_model_across_instances_with_same_key(monkeypatch):
    """2026-09-13 resource constraint: bench's V4Retriever.reset() creates a
    fresh Embedder per conversation (up to 150x for LongMemEval oracle) --
    without a process-wide model cache this reloaded (and kept resident
    until GC) a full copy of the model each time, which is what triggered a
    real `MPS backend out of memory` crash for the granite embedder. Two
    Embedder instances with the same (model_name, device, max_seq_length)
    must share the exact same underlying model object."""
    import sentence_transformers

    from memory_os_v4 import embed as embed_mod

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(embed_mod, "_MODEL_CACHE", {})  # isolate from other tests/process state
    _FakeSentenceTransformer.instances.clear()

    e1 = Embedder("shared/model", device="cpu", max_seq_length=256)
    e2 = Embedder("shared/model", device="cpu", max_seq_length=256)
    e1._ensure_model()
    e2._ensure_model()
    assert e1._model is e2._model
    assert len(_FakeSentenceTransformer.instances) == 1  # loaded once, not twice


def test_embedder_does_not_share_model_across_different_max_seq_length(monkeypatch):
    import sentence_transformers

    from memory_os_v4 import embed as embed_mod

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(embed_mod, "_MODEL_CACHE", {})
    _FakeSentenceTransformer.instances.clear()

    e1 = Embedder("shared/model", device="cpu", max_seq_length=256)
    e2 = Embedder("shared/model", device="cpu", max_seq_length=512)
    e1._ensure_model()
    e2._ensure_model()
    assert e1._model is not e2._model
    assert len(_FakeSentenceTransformer.instances) == 2


def test_resolve_device_explicit_passthrough():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("mps") == "mps"


def test_hash_embedder_deterministic_and_normalized():
    e = HashEmbedder(dim=32)
    v1 = e.encode(["hello world"])
    v2 = e.encode(["hello world"])
    assert np.allclose(v1, v2)
    norm = float(np.linalg.norm(v1[0]))
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_hash_embedder_empty_text_returns_zero_vector_without_crashing():
    e = HashEmbedder(dim=16)
    v = e.encode([""])
    assert v.shape == (1, 16)
