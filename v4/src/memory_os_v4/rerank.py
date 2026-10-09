"""Optional cross-encoder reranker. Lazy-loaded; degrades silently if the
configured model is not locally available (logs once, callers proceed without
reranking)."""

from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger("memory_os_v4")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch always present here
        return "cpu"


class CrossEncoderReranker:
    """Wraps a sentence-transformers ``CrossEncoder``; never raises.

    ``device="auto"`` (default) resolves to mps on Apple Silicon, else cpu
    (Palier 2-3 E11: `tomaarsen/ms-marco-ettin-17m-reranker` is cached and
    verified working on both). If scoring on the resolved device ever comes
    back with a non-finite value (NaN/inf -- e.g. the deleted
    `cross-encoder/ms-marco-MiniLM-L6-v2`, which produced NaN with
    transformers 5.2 on mps), the reranker falls back to cpu once and
    retries the same batch rather than poisoning the fused ranking with a
    NaN cross-encoder score.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._failed = False
        self._warned = False
        self._cpu_fallback_done = False

    def _load(self, device: str):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, device=device)

    def _ensure(self):
        if self._model is None and not self._failed:
            try:
                self._model = self._load(_resolve_device(self.device))
            except Exception as exc:  # noqa: BLE001 - any load failure is non-fatal
                self._failed = True
                if not self._warned:
                    logger.warning(
                        "cross-encoder %s unavailable (%s); continuing without rerank",
                        self.model_name,
                        exc,
                    )
                    self._warned = True
        return self._model

    def available(self) -> bool:
        return self._ensure() is not None

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        preds = self._model.predict(pairs)
        return [float(x) for x in preds]

    def score(self, query: str, texts: list[str]) -> list[float] | None:
        model = self._ensure()
        if model is None or not texts:
            return None
        # Score the item text directly, never the (query, header+text)
        # concatenation used for dense/BM25 indexing (memory.py
        # `contextual_header`) -- callers pass raw `item.text` here.
        pairs = [(query, t) for t in texts]
        try:
            scores = self._predict(pairs)
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            logger.warning("cross-encoder inference failed (%s); continuing without rerank", exc)
            return None
        if not all(math.isfinite(s) for s in scores) and not self._cpu_fallback_done:
            self._cpu_fallback_done = True
            logger.warning(
                "cross-encoder %s produced non-finite score(s) on device=%r; retrying on cpu",
                self.model_name,
                _resolve_device(self.device),
            )
            try:
                self._model = self._load("cpu")
                scores = self._predict(pairs)
            except Exception as exc:  # noqa: BLE001
                self._failed = True
                logger.warning("cross-encoder cpu fallback failed (%s); continuing without rerank", exc)
                return None
            if not all(math.isfinite(s) for s in scores):
                self._failed = True
                logger.warning(
                    "cross-encoder %s still non-finite on cpu; continuing without rerank", self.model_name
                )
                return None
        return scores
