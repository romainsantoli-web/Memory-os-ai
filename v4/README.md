# Memory OS v4

Local-first long-term memory engine for LLM agents. Clean rewrite under `v4/src/memory_os_v4`;
does not import the legacy `memory_os_ai` package. See `docs/ARCHITECTURE.md` for the binding
interface contract and `docs/RESEARCH_NOTES.md` for the rationale behind each mechanism.

## Install / run

No new dependencies. Use the parent repo's venv:

```bash
cd v4
PYTHONPATH=src /Users/romain/Memory-os-ai/.venv/bin/python -m pytest tests -q
```

```python
from memory_os_v4 import MemoryV4, Config

m = MemoryV4(config=Config(data_dir="~/.memory-os-v4"))  # defaults to MiniLM
m.add_episode("We picked FAISS FlatL2 for the vector index.", project="prometheus")
m.add_facts([{"text": "Romain lives in Paris."}], project="prometheus")
hits = m.search("vector index choice", top_k=5)
brief = m.session_brief(focus_query="prometheus")
m.flush()
```

For fast, ML-free tests or scripting, select the hashing embedder: `Config(embed_model="hash")`.

## Package layout

| module | responsibility |
|---|---|
| `config.py` | `Config` dataclass, `MEMORY_V4_*` env overrides |
| `store.py` | `MemoryItem`, append-only `items.jsonl` log, embedding cache I/O |
| `text.py` | tokenize, sentence split, sentence-aware chunking, contextual header |
| `bm25.py` | Okapi BM25 (k1=1.2, b=0.75), incremental, JSON-persisted |
| `embed.py` | `Embedder` (sentence-transformers, MPS/CPU auto) + `HashEmbedder` (no ML) |
| `vector_index.py` | FAISS `IndexFlatIP`/`IndexIDMap2` wrapper (lazy faiss import — see below) |
| `entities.py` | heuristic entity extraction + inverted-index `EntityIndex` |
| `temporal.py` | EN/FR absolute + relative date parsing, validity/recency helpers |
| `rerank.py` | optional lazy `CrossEncoderReranker` |
| `retrieval.py` | `HybridRetriever` (dense+BM25+entity fusion), `mmr()`, `SearchHit` |
| `extract.py` | client-fact validation, heuristic fact extraction |
| `llm.py` | minimal OpenAI-compatible client (urllib), off unless configured |
| `blocks.py` | Letta-style core memory blocks |
| `consolidate.py` | dedup / supersession sweeps, decay, summary-cluster proposals |
| `memory.py` | `MemoryV4` façade tying everything together |
| `migrate.py` | v3 `_conversation_log.jsonl` reader (idempotent) |
| `models.py` | pydantic input models for the MCP tool surface |
| `server.py` | MCP stdio server (`python -m memory_os_v4.server`) — see "MCP server" below |
| `instructions.py` | `MEMORY_INSTRUCTIONS` injected into every MCP session |

## MCP server

`server.py` exposes the engine over MCP (stdio transport) so any MCP-compatible client (Claude
Code, Claude Desktop, Codex CLI, etc.) can use it as a memory tool. Install-free: it reuses the
parent repo's venv, no new packages.

### Setup

Copy `v4/mcp.example.json`'s `memory-os-v4` entry into your client's MCP config (e.g.
`~/.claude.json` or `claude_desktop_config.json`), or run directly:

```bash
PYTHONPATH=/Users/romain/Memory-os-ai/v4/src MEMORY_V4_DIR=~/.memory-os-v4 \
  HF_HUB_OFFLINE=1 /Users/romain/Memory-os-ai/.venv/bin/python -m memory_os_v4.server
```

Env vars: `MEMORY_V4_DIR` (data dir, default `~/.memory-os-v4`), `MEMORY_V4_EMBED_MODEL` (default
MiniLM; use `hash` for ML-free runs), `MEMORY_V4_PROJECT` (default project label substituted for
any tool call that omits `project`/`project_label`, default `"default"`), `MEMORY_V4_DEVICE`,
`MEMORY_V4_RERANK_MODEL`, `MEMORY_V4_LLM_URL`/`MEMORY_V4_LLM_MODEL`, `MEMORY_WORKSPACE` (extra
allowed root for `memory_ingest`/`memory_migrate_v3`, alongside `$HOME`).

### Selfcheck

Exercises every tool once against a throwaway hash-embedder engine in a temp dir and exits
non-zero on any error:

```bash
PYTHONPATH=/Users/romain/Memory-os-ai/v4/src /Users/romain/Memory-os-ai/.venv/bin/python \
  -m memory_os_v4.server --selfcheck
```

### Tools

Compat names (same names/required params as the v3 MCP server, richer output): `memory_status`,
`memory_session_brief`, `memory_chat_save`, `memory_search`, `memory_get_context`,
`memory_ingest`, `memory_list_documents`, `memory_search_occurrences`, `memory_compact` (alias of
`memory_consolidate`). New in v4: `memory_add_facts`, `memory_timeline`, `memory_block_get`,
`memory_block_set`, `memory_consolidate`, `memory_forget`, `memory_migrate_v3`. Every tool
validates its input against a pydantic model in `models.py` (`extra="forbid"`, so a typo'd
argument is rejected rather than silently ignored) and returns a JSON-serializable dict; on error
it returns `{"ok": false, "error": "<message>"}` instead of raising — never a traceback to the
client. Timestamps in output are ISO-8601 strings; floats are rounded to 4 decimals. See
`docs/ARCHITECTURE.md`'s "MCP server v4 — tool surface" table for the full per-tool input/output
contract.

### How fact extraction works

`instructions.py` (`MEMORY_INSTRUCTIONS`, injected as the server's `Server(instructions=...)`)
teaches the calling model to: recall via `memory_session_brief` at session start; save via
`memory_chat_save` regularly, passing atomic, self-contained, **ADD-only** facts (one sentence,
explicit subject, no pronouns, specific entities — "Nisha's dad" not "dad") with absolute dates;
never ask to delete a fact — a changed fact is added anew and the server supersedes the old one
deterministically (same/overlapping entities, cosine similarity ≥ `config.supersede_cos`, not a
near-duplicate); tag tasks `["todo"]`; record lessons from both successes and failures as
`kind="procedure"`; and record its own actions as facts too ("Claude created ~/x/y.py on
2026-09-12"). `memory_forget` is a deliberate escape hatch for the rare true deletion request
(soft-delete only, by `item_id` or by `query` + `confirm=true`) — it is never used for updates.

### Keywords and multi-query search (production port of the bench gains, 2026-09-14)

Two LoCoMo-validated bench mechanisms (`bench/retrievers.py`'s `V4EnrichedRetriever`
"facts+keywords" mode and `V4PlannedRetriever` RRF query planning) are now in the production
engine, both backward-compatible: default behaviour is unchanged for every existing caller that
doesn't pass the new optional fields.

- **Keywords**: `memory_chat_save` messages and facts (`FactIn`, via `memory_add_facts` or
  `memory_chat_save(facts=...)`) accept an optional `keywords` list (3-8 lowercase topical words —
  concrete nouns, activities, places, names). They are appended to the DENSE/BM25 indexed text
  (`memory.index_text()`, one helper used everywhere that text is built; gated by
  `Config.index_keywords`, default on) — never to the stored/displayed `text`, and never to
  `tags`/`entities`. `instructions.py` now asks the calling model to add them at save time.
- **Multi-query search (RRF fusion)**: `memory_search` and `memory_get_context` accept optional
  `queries` (2-4 short sub-questions) and `hypotheses` (1-2 HyDE-style sentences that would answer
  the question). When either is given, the engine searches the original query and every extra
  string independently (cross-encoder rerank disabled per sub-search, `top_k * 4` candidates
  each), fuses the ranked hit lists via Reciprocal Rank Fusion (`Config.multi_query_rrf_k`,
  default 20; weights: original query 1.0, `queries` 1.0, `hypotheses` 0.7 —
  `HybridRetriever.search_multi`), then applies the usual optional cross-encoder rerank on the
  fused top `rerank_top_n`. `instructions.py` asks the calling model to decompose
  comparison/counting/multi-part questions into `queries` and propose `hypotheses`. Guard: at
  most 8 extra strings combined (`queries` first, then `hypotheses`), 300 characters each,
  silently clipped rather than rejected.

### Facet channel (production port of the bench gain, 2026-09-19, OFF by default)

A third LoCoMo-validated bench mechanism (`bench/retrievers.py`'s `V4PlannedRetriever` facet
branch, `build_facet_index`/`rank_facet_pool`) is now in the production engine, gated behind
`Config.facet_channel` (env `MEMORY_V4_FACET_CHANNEL`, default **off**) since it was only measured
on LoCoMo so far (R@10 0.721 → 0.757 with `facet_weight=2.0`, no reranker — see `facets.py` and
`config.py`). Facts/messages (`FactIn`/`MessageIn`, via `memory_add_facts`/`memory_chat_save`)
accept an optional `facets` list of `{subject, predicate, object}` triples, `predicate` drawn from
a closed 43-item vocabulary (`memory_os_v4.facets.PREDICATES`, mirrored from
`bench/data/enrich/PREDICATES.md`; anything else normalizes to `"other"`, which is stored but
never indexed). They are normalized (`facets.normalize_facet`) and indexed by an in-memory
`facets.FacetIndex` (`(subject, predicate) -> {item_id}`, derived state rebuilt from `items.jsonl`
at load, never its own on-disk file) **regardless of the flag**, so storing costs nothing and
turning the channel on later needs no re-index. `memory_search`/`memory_get_context` accept an
optional `facets` list of `{subject, predicate}` query patterns (no `object`); when the flag is on
and the resulting pool is non-empty, `HybridRetriever.search_multi` fuses the pool — ranked by
each member's best rank across the query/sub-query/hypothesis hit lists (`facets.rank_facet_pool`)
— as one more weighted party in the RRF combination, even when no `queries`/`hypotheses` were
given. With the flag off, or no `facets` given, or an empty pool, behaviour is byte-identical to
today. `instructions.py` only mentions the facet channel (predicates included) when the flag is on.

### Migrating from v3

```bash
PYTHONPATH=/Users/romain/Memory-os-ai/v4/src /Users/romain/Memory-os-ai/.venv/bin/python -c "
from memory_os_v4 import MemoryV4, Config
m = MemoryV4(config=Config(data_dir='~/.memory-os-v4'))
print(m.migrate_v3('~/.memory-os-ai/_conversation_log.jsonl'))
m.flush()
"
```

Or, once the server is running, call the `memory_migrate_v3` tool (defaults to that same path,
read-only on it, idempotent — safe to re-run).

## Design clarifications (not deviations from ARCHITECTURE.md, but worth stating explicitly)

- **`valid_from` defaults to `event_time`** when a caller doesn't supply one (for episodes, facts,
  and docs alike). This is what makes bi-temporal `at_time` queries behave correctly: without it,
  a fact would appear "always valid" even before it was recorded, and the supersession example in
  the spec (a later fact excluding an earlier one from `at_time` queries before it existed) would
  not work. `MemoryItem`'s own default (`valid_from=None`, i.e. open) is unchanged for callers who
  construct items directly.
- **Single-sentence entity extraction keeps the sentence-start subject.** The literal spec rule
  ("capitalized spans not at sentence start, unless also seen elsewhere") is applied only for
  multi-sentence text. A one-sentence atomic fact ("Romain lives in Paris.") has no "elsewhere" to
  corroborate against, so dropping its subject would break entity-overlap supersession for the most
  common case (atomic facts). Multi-sentence documents/episodes still use the full start/mid rule.
- **`faiss` is imported lazily**, inside `VectorIndex.__init__`, not at module load time (see next
  section). `memory_os_v4/__init__.py` also pins `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
  `VECLIB_MAXIMUM_THREADS`/`OPENBLAS_NUM_THREADS`/`KMP_DUPLICATE_LIB_OK` to avoid it.

## Known limitation: faiss + torch segfault on this machine

On this Apple Silicon build (faiss-cpu 1.13.2 + torch 2.10 MPS build), importing `faiss` before
sentence-transformers has run at least one real forward pass **reliably segfaults the process**
the first time both a torch computation and a faiss search happen in the same process (two
independent multi-threaded BLAS/OpenMP runtimes racing). Reproduced deterministically outside this
package too. Mitigation implemented here:

1. `faiss` is imported lazily (`VectorIndex.__init__`), so merely importing `memory_os_v4` doesn't
   pull it in ahead of the embedder.
2. `Embedder._ensure_model()` runs a one-token warm-up `encode()` right after loading the model,
   before any `VectorIndex` (and therefore `faiss`) is constructed.
3. Both libraries are pinned to a single BLAS thread (`torch.set_num_threads(1)`,
   `faiss.omp_set_num_threads(1)`, plus the process-wide env vars in `__init__.py`).

With all three in place the real-MiniLM integration test (`test_core_slow_integration.py`) runs
clean. This is an environment workaround, not a library bug fix — if faiss or torch wheels change,
re-verify with a two-line repro (`import faiss; SentenceTransformer(...).encode(...)`) before
relaxing any of the three mitigations. Single-threaded BLAS is a acceptable trade for a memory
store at this scale (< 100k vectors, FlatIP is brute-force regardless of thread count).

## Other known limitations

- `add_facts` dedup/supersession scans are O(n) over all live facts per new fact (fine at the
  tested scale; `bench/` measures the hybrid retriever itself at larger corpora separately).
- `HashEmbedder` is a hashing bag-of-words embedder for fast, ML-free tests — it has no semantic
  generalization, so paraphrase-level supersession (different wording, same meaning) only works
  reliably with the real sentence-transformers embedder. Tests that need that use either literal
  wording overlap or a per-test lower `supersede_cos` with `HashEmbedder`, or the real embedder.
- `llm.py`/`extract.llm_facts` are inert unless `MEMORY_V4_LLM_URL` is set to a reachable
  OpenAI-compatible server; never required by tests.

## Results (2026-09-13, model-free evidence recall, `PYTHONHASHSEED=0`)

Turn-level Recall@10 of the gold evidence, macro over LoCoMo categories 1-4 (1 536 questions) and
over the 6 LongMemEval-oracle question types (stratified 150). Full ledger: `RESULTS.md`; method:
`bench/README.md`; ablations and hardening log: `docs/ABLATIONS.md`.

| system | LoCoMo R@10 | LoCoMo MRR | LongMemEval R@10 | p50 search |
|---|---|---|---|---|
| v3 exact replica (128-char windows, FlatL2, L2 threshold 0.8) | 0.175 | — | — | — |
| v3-style charitable baseline (same + chunk→turn dedup, no threshold) | 0.405 | 0.279 | — | 15.6 ms |
| BM25 over whole turns | 0.424 | 0.304 | 0.872 | 0.1 ms |
| v4 first default (RRF, entities on) | 0.435 | 0.293 | 0.776 | 7.9 ms |
| v4 with MiniLM (cc fusion dense 1.0 / BM25 1.5, stemming, candidates 100) | 0.527 | 0.398 | 0.832 | 26.5 ms |
| **v4 default now** (same + `granite-embedding-97m-multilingual-r2`, 210 MB) | **0.538** | — | **0.837** | 18-30 ms |
| v4 + Ettin-17m reranker top-30 (opt-in, `MEMORY_V4_RERANK_MODEL`) | 0.565 | — | 0.827 | 166 ms |
| v4 + LLM-extracted facts and keywords (Sonnet 5 offline, bench-only test of the production `memory_chat_save(facts=…)` mechanism) | 0.620 | 0.459 | — | 32 ms |
| v4 + LLM facts/keywords + Ettin-17m reranker top-30 | 0.662 | 0.541 | — | 142 ms |
| v4 + LLM facts/keywords + LLM query planning (sub-queries + hypotheses, RRF k=20) + Ettin-68m reranker | 0.751 | 0.65 | — | 1.1 s (bench-only) |
| v4 + all of the above + LLM listwise reranking of the top-20 (Sonnet 5, offline) | 0.777 | 0.762 | — | bench-only |
| v4 + LLM listwise reranking of the top-50 + ensemble with the top-20 ranking | **0.815** | **0.773** | — | bench-only |

**Production status**: the keywords (`0.538 → 0.620` row) and query-planning (`0.662 → 0.751`
row) mechanisms above are now implemented in the production engine itself (`MemoryItem.keywords`
+ `memory.index_text()`; `HybridRetriever.search_multi` RRF fusion) and exposed on
`memory_chat_save`/`memory_add_facts` (`keywords`) and `memory_search`/`memory_get_context`
(`queries`, `hypotheses`) — see "MCP server" above. The bench rows used an offline LLM to extract
keywords/facts and plan sub-queries/hypotheses; production instead asks the calling MCP client's
own model to supply them at save/search time — same retrieval-side mechanism, different (online,
free) source for the enrichment.

Known weakness: LongMemEval `temporal-reasoning` (0.51 vs 0.87 for BM25 alone); recency, BM25
weight and a temporal-query mode were tested and refuted (see `docs/ABLATIONS.md`). Independent
harness reference for the same metric (A-Mem, arXiv:2606.24775): LoCoMo R@10 0.859; the recall ceiling of
v4's 50-candidate pool is 0.870, so the remaining gap is LLM reranking precision on multi-hop and
open-domain questions — the remaining gap is
LLM-side enrichment of memories (keywords/context per note), which the bench cannot measure but which
the MCP protocol provides in production via client-extracted facts. Entity-based multi-hop expansion
was tested and rejected (net negative); `cross-encoder/ms-marco-MiniLM-L6-v2` produces NaN with
transformers 5.2 — use `tomaarsen/ms-marco-ettin-17m-reranker` (68 MB) instead. Disk footprint of
all models: 365 MB.

## Switching a client from v3 to v4

The v3 server, its `.mcp.json` and data (`~/.memory-os-ai`) are untouched. To try v4 in Claude Code,
add the `memory-os-v4` entry from `v4/mcp.example.json` to your MCP config (keep or drop the v3
entry), then import the v3 history once:

```bash
cd /Users/romain/Memory-os-ai/v4 && PYTHONPATH=src HF_HUB_OFFLINE=1 MEMORY_V4_DIR=~/.memory-os-v4 /Users/romain/Memory-os-ai/.venv/bin/python -c "from memory_os_v4 import MemoryV4; m=MemoryV4(); print(m.migrate_v3('~/.memory-os-ai/_conversation_log.jsonl')); m.flush()"
```

Self-check of every MCP tool (temp dir, no model download):

```bash
cd /Users/romain/Memory-os-ai/v4 && PYTHONPATH=src HF_HUB_OFFLINE=1 /Users/romain/Memory-os-ai/.venv/bin/python -m memory_os_v4.server --selfcheck
```
