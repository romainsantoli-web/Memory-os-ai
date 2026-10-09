"""CrossEncoderReranker unit tests (network-free: a fake CrossEncoder stands
in for the real model -- see docs/ABLATIONS.md Palier 2-3 E11)."""

from __future__ import annotations

from memory_os_v4.rerank import CrossEncoderReranker, _resolve_device


class _FakeCrossEncoder:
    """Records the device it was constructed with; ``predict`` returns a
    fixed score list, or raises/returns NaN depending on the test."""

    def __init__(self, model_name: str, device: str = "cpu", scores=None, raise_on_predict=False):
        self.model_name = model_name
        self.device = device
        self._scores = scores if scores is not None else [1.0] * 999
        self._raise = raise_on_predict

    def predict(self, pairs):
        if self._raise:
            raise RuntimeError("boom")
        return self._scores[: len(pairs)]


def _install_fake_ce(monkeypatch, **kwargs):
    import sentence_transformers

    def factory(model_name, device="cpu"):
        return _FakeCrossEncoder(model_name, device, **kwargs)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", factory)


def test_resolve_device_explicit_passthrough():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("mps") == "mps"


def test_resolve_device_auto_uses_mps_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"


def test_reranker_loads_on_resolved_device(monkeypatch):
    _install_fake_ce(monkeypatch, scores=[1.0, 2.0])
    r = CrossEncoderReranker("fake/model", device="cpu")
    assert r.available()
    assert r._model.device == "cpu"


def test_reranker_scores_query_item_text_pairs_directly(monkeypatch):
    """E11 spec: score (query, item.text) -- never a header-duplicated
    concatenation. This test only asserts the reranker scores whatever text
    list it is given (the header-exclusion itself is exercised end-to-end in
    test_core_retrieval.py's rerank tests, which pass raw item.text)."""
    _install_fake_ce(monkeypatch, scores=[5.0, -3.9])
    r = CrossEncoderReranker("fake/model", device="cpu")
    out = r.score("q", ["relevant text", "irrelevant text"])
    assert out == [5.0, -3.9]


def test_reranker_score_returns_none_for_empty_texts(monkeypatch):
    _install_fake_ce(monkeypatch)
    r = CrossEncoderReranker("fake/model", device="cpu")
    assert r.score("q", []) is None


def test_reranker_unavailable_model_never_raises(monkeypatch):
    import sentence_transformers

    def boom(model_name, device="cpu"):
        raise OSError("not cached")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", boom)
    r = CrossEncoderReranker("missing/model")
    assert r.available() is False
    assert r.score("q", ["a"]) is None


def test_reranker_predict_failure_never_raises(monkeypatch):
    _install_fake_ce(monkeypatch, raise_on_predict=True)
    r = CrossEncoderReranker("fake/model", device="cpu")
    assert r.score("q", ["a"]) is None


def test_reranker_falls_back_to_cpu_on_nan(monkeypatch):
    """The deleted cross-encoder/ms-marco-MiniLM-L6-v2 produced NaN with
    transformers 5.2 on this machine -- verify the generic (model-agnostic)
    fallback: a non-finite score on the resolved device triggers one cpu
    retry rather than poisoning the fused ranking with NaN."""
    import sentence_transformers

    calls = []

    def factory(model_name, device="cpu"):
        calls.append(device)
        scores = [float("nan"), 1.0] if device == "mps" else [0.7, 1.0]
        return _FakeCrossEncoder(model_name, device, scores=scores)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", factory)
    r = CrossEncoderReranker("fake/model", device="mps")
    out = r.score("q", ["a", "b"])
    assert out == [0.7, 1.0]
    assert calls == ["mps", "cpu"]  # loaded once, then retried on cpu


def test_reranker_still_nan_on_cpu_gives_up_gracefully(monkeypatch):
    _install_fake_ce(monkeypatch, scores=[float("nan")])
    r = CrossEncoderReranker("fake/model", device="cpu")
    assert r.score("q", ["a"]) is None
    assert r._failed is True
