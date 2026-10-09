"""Evaluation bench for Memory OS v4.

Pure-evaluation package: loads LoCoMo / LongMemEval(-oracle) datasets, wraps
several retrievers (a faithful reproduction of the old v3 chunking scheme,
plain dense/BM25 baselines, and the v4 core when available) behind one
`Retriever` protocol, scores them with recall/MRR/nDCG, and appends rows to
the project-level RESULTS.md ledger.

Never imports memory_os_v3 or memory_os_v4 internals beyond the public
`MemoryV4` façade documented in docs/ARCHITECTURE.md.
"""
