"""FAISS-backed dense vector index (IndexFlatIP + IndexIDMap2).

String ids are mapped to int64 faiss ids via a truncated sha1 hash, with linear
probing on collision. Vectors are also kept in a plain dict so callers (MMR,
supersession cosine checks) can fetch them in O(1) without a faiss round-trip,
and so the index can be exported for persistence (``Store`` owns the actual
``embeddings.npy`` / ``ids.json`` files; this module hands it the arrays).

``faiss`` is imported lazily (inside :meth:`VectorIndex.__init__`) rather than
at module load time. On macOS, importing faiss before sentence-transformers
has run its first forward pass reliably segfaults the process (two conflicting
OpenMP runtimes race once torch starts its own threads) -- deferring the
import lets ``MemoryV4`` load the embedder first, which sidesteps the crash.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _assign_int_id(sid: str, taken: dict[int, str]) -> int:
    digest = hashlib.sha1(sid.encode("utf-8")).digest()[:8]
    iid = int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF
    while iid in taken and taken[iid] != sid:
        iid = (iid + 1) & 0x7FFFFFFFFFFFFFFF
    return iid


class VectorIndex:
    """Dense similarity search over normalized embeddings (inner product = cosine)."""

    def __init__(self, dim: int) -> None:
        import faiss  # local: see module docstring for why this must stay lazy

        faiss.omp_set_num_threads(1)  # avoids a faiss/torch segfault, see module docstring
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._vecs: dict[str, np.ndarray] = {}
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def add(self, ids: list[str], vecs: np.ndarray) -> None:
        if len(ids) == 0:
            return
        # A4 (adversarial review, 2026-09-12): de-duplicate ids WITHIN this
        # single call first (first occurrence wins) -- a duplicated id in one
        # batch (e.g. migrate_v3 replaying a duplicated log line) used to add
        # 2 vectors mapped to the same int id for 1 stored item, since the
        # `existing` check below only catches ids already indexed by a PRIOR
        # call, not duplicates inside this one.
        seen: set[str] = set()
        keep_idx: list[int] = []
        for i, sid in enumerate(ids):
            if sid in seen:
                continue
            seen.add(sid)
            keep_idx.append(i)
        if len(keep_idx) != len(ids):
            ids = [ids[i] for i in keep_idx]
            vecs = vecs[keep_idx]
        existing = [sid for sid in ids if sid in self._str_to_int]
        if existing:
            self.remove(existing)
        vecs = np.ascontiguousarray(vecs.astype(np.float32))
        int_ids = []
        for sid in ids:
            iid = _assign_int_id(sid, self._int_to_str)
            self._str_to_int[sid] = iid
            self._int_to_str[iid] = sid
            int_ids.append(iid)
        self._index.add_with_ids(vecs, np.array(int_ids, dtype=np.int64))
        for sid, vec in zip(ids, vecs):
            self._vecs[sid] = vec

    def ids(self) -> list[str]:
        """All string ids currently indexed (item ids and, if multi_vector is
        on, chunk ids alike). Used by ``MemoryV4._reconcile_indexes`` (A1) to
        diff the vector index against the live store without reaching into
        the private ``_vecs``/``_str_to_int`` maps."""
        return list(self._vecs.keys())

    def remove(self, ids: list[str]) -> None:
        int_ids = []
        for sid in ids:
            iid = self._str_to_int.pop(sid, None)
            if iid is not None:
                self._int_to_str.pop(iid, None)
                int_ids.append(iid)
            self._vecs.pop(sid, None)
        if int_ids:
            self._index.remove_ids(np.array(int_ids, dtype=np.int64))

    def search(self, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.ntotal == 0 or top_k <= 0:
            return []
        query = np.ascontiguousarray(vec.reshape(1, -1).astype(np.float32))
        scores, ids = self._index.search(query, min(top_k, self.ntotal))
        out: list[tuple[str, float]] = []
        for score, iid in zip(scores[0], ids[0]):
            if iid == -1:
                continue
            sid = self._int_to_str.get(int(iid))
            if sid is not None:
                out.append((sid, float(score)))
        return out

    def get_one(self, sid: str) -> np.ndarray | None:
        return self._vecs.get(sid)

    def get(self, ids: list[str]) -> dict[str, np.ndarray]:
        return {sid: self._vecs[sid] for sid in ids if sid in self._vecs}

    def scores_for(self, vec: np.ndarray, ids: list[str]) -> dict[str, float]:
        """True (not top-k-limited) inner-product score of `vec` against
        each of `ids`'s stored vectors -- the same value `search()` would
        return for an id within its top-k, but computed for an arbitrary id
        list regardless of whether faiss's own top-k search would surface
        it (used by ``Config.full_candidate_scoring``, see retrieval.py).
        Vectors are unit-normalized at encode time (see embed.py), so this
        inner product equals cosine similarity, exactly like `search()`'s
        IndexFlatIP score. An id with no stored vector scores 0.0.
        """
        if not ids:
            return {}
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        out: dict[str, float] = {}
        for sid in ids:
            v = self._vecs.get(sid)
            out[sid] = float(np.dot(q, v)) if v is not None else 0.0
        return out

    def export(self) -> tuple[list[str], np.ndarray]:
        """Return (ids, vecs) in a stable order, for cache persistence."""
        ids = list(self._vecs.keys())
        if not ids:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.stack([self._vecs[sid] for sid in ids]).astype(np.float32)
        return ids, vecs
