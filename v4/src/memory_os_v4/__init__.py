"""Memory OS v4: a local-first long-term memory engine for LLM agents."""

import os as _os

# On macOS, faiss (its own bundled OpenMP) and torch/sentence-transformers
# (Accelerate/MKL) each spin up a multi-threaded BLAS pool; running both
# multi-threaded in one process reliably segfaults during faiss search once
# torch has also run. Forcing single-threaded BLAS everywhere avoids the
# conflict entirely; set before either library is ever imported (see also the
# defensive per-library calls in embed.py and vector_index.py).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .config import Config  # noqa: E402
from .memory import MemoryV4  # noqa: E402
from .retrieval import SearchHit  # noqa: E402
from .store import MemoryItem  # noqa: E402

__all__ = ["MemoryV4", "Config", "MemoryItem", "SearchHit"]

__version__ = "4.0.0"
