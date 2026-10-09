# Memory OS v4 — Architecture & interface spec

Package: `memory_os_v4` (under `v4/src`). Python 3.11. Runtime deps = those already in `../.venv`
(numpy, faiss-cpu, sentence-transformers, torch, mcp, pydantic). No new heavy deps: BM25, RRF,
entity extraction, date parsing are implemented in-house.

## Modules
| module | responsibility |
|---|---|
| `config.py` | `Config` dataclass from env (`MEMORY_V4_DIR`, `MEMORY_V4_EMBED_MODEL`, `MEMORY_V4_RERANK_MODEL`, weights, decay) |
| `store.py` | `MemoryItem` dataclass, append-only JSONL log (`items.jsonl`), tombstones/updates as new lines, `load()` replays the log; `embeddings.npy` + `ids.json` cache |
| `text.py` | `tokenize()` (lowercase, unicode-aware, keeps code identifiers, strips accents for matching), `split_sentences()`, `chunk_turns()` (turn = one chunk; long turns split on sentences ≤ `max_tokens` with overlap), `contextual_header(item)` |
| `bm25.py` | in-house BM25 (k1=1.2, b=0.75), incremental `add(doc_id, tokens)`, `search(query_tokens, top_k)`, persisted as JSON |
| `embed.py` | `Embedder` (sentence-transformers, MPS/CPU auto, normalize, batch, model configurable, lazy) |
| `vector_index.py` | FAISS `IndexFlatIP` wrapped with `IndexIDMap2`; `add(ids, vecs)`, `remove(ids)`, `search(vec, top_k)`; save/load |
| `entities.py` | heuristic extractor: capitalized spans, `Code_Identifiers`, quoted terms, numbers+units, dates; inverted index entity→ids |
| `facets.py` | closed-predicate-vocabulary facet channel (production port, 2026-09-19): `normalize_facet`/`normalize_facets`, in-memory `FacetIndex` ((subject, predicate)→ids, derived state, no on-disk file), `rank_facet_pool` |
| `temporal.py` | parse absolute/relative dates in text given `event_time`; `is_valid_at(item, t)`; `recency_weight(age_days, half_life_days)` |
| `retrieval.py` | `HybridRetriever.search(query, top_k, at_time=None, kinds=None, project=None)` → dense + bm25 + entity candidates → RRF (k=60) or weighted-norm fusion → × recency/importance → optional cross-encoder rerank of top-N → MMR |
| `rerank.py` | optional `CrossEncoderReranker` (lazy; returns None if model missing) |
| `extract.py` | `facts_from_client(facts)` (validate/normalise), `facts_heuristic(text)` (declarative sentences with entities/numbers), `facts_llm(text)` via `llm.py` if configured |
| `llm.py` | optional OpenAI-compatible client (llama-server / Ollama / LM Studio); `available()` false by default |
| `blocks.py` | Letta-style core blocks: project-scoped `get`/`set`, explicit global sharing, char limit, scoped history in `blocks.jsonl` |
| `consolidate.py` | project-scoped active dedup (facts/beliefs require conservative assertion equality), explicit replacement/identity sweep, decay/reinforce, summary proposals (clusters) |
| `memory.py` | façade `MemoryV4`: `add_episode()`, `add_facts()`, `add_document()`, `search()`, `get_context()`, `session_brief()`, `consolidate()`, `status()`, `migrate_v3()` |
| `server.py` | MCP stdio server: v3 tool names kept (`memory_session_brief`, `memory_chat_save`, `memory_search`, `memory_get_context`, `memory_status`, `memory_compact`→consolidate, `memory_ingest`, `memory_list_documents`) + new (`memory_add_facts`, `memory_block_get`, `memory_block_set`, `memory_consolidate`, `memory_timeline`, `memory_forget`) |
| `instructions.py` | MCP server instructions: recall at start, **extract atomic ADD-only facts at save**, temporal phrasing, agent facts first-class |
| `migrate.py` | read v3 `_conversation_log.jsonl` → episodes (+summary items) |

## Data model — `MemoryItem`
```
id: str (uuid4 hex)          kind: "episode"|"fact"|"summary"|"procedure"|"doc"|"belief"
text: str                    role: "user"|"assistant"|"system"|"doc"
project: str                 source: "chat_save"|"ingest"|"consolidate"|"migrate_v3"|"client"
doc_ref: str|None            parent_ids: list[str]   (provenance: episodes a fact came from)
created_at: float (unix, ingestion)      event_time: float (when it happened; default created_at)
valid_from: float|None       valid_until: float|None     invalidated_by: str|None
entities: list[str]          tags: list[str]             importance: float (0-1, default 0.5)
keywords: list[str]          facets: list[dict]          confidence: float (0-1)
access_count: int            last_accessed: float|None   hash: str (sha1 of normalised text)
deleted: bool
```
`keywords` (2026-09-14) and `facets` (2026-09-19) are both backward-compatible production ports of
bench-validated mechanisms: absent in an old `items.jsonl` record, `MemoryItem.from_dict` leaves
them at their default (`[]`). `facets` entries are `facets.normalize_facet`'s output
(`{"subject", "subject_key", "predicate", "object"}`); `predicate` is one of `facets.PREDICATES`
(closed vocabulary) or `"other"`. Unlike `keywords`, `facets` are NEVER appended to `index_text` --
they only feed the separate `facets.FacetIndex` (see "Facet channel" below), so dense/BM25 recall
is completely unaffected by this field.
Log semantics: every mutation appends a full record; `load()` keeps the last record per id.

## Retrieval scoring (default config)
```
cand = union(dense top 50, bm25 top 50, entity top 50)
rrf(i) = Σ_signals w_s / (60 + rank_s(i))        w_dense=1.0, w_bm25=1.0, w_entity=0.5
score(i) = rrf(i) × (0.7 + 0.3·recency(i)) × (0.8 + 0.4·importance(i))
recency = exp(-ln2 · age_days / half_life)       half_life default 30 d (episodes), 180 d (facts)
validity filter when at_time given: valid_from ≤ at_time < valid_until (None = open)
rerank (optional): cross-encoder on top 30, final = 0.5·norm(score) + 0.5·norm(ce)
MMR λ=0.7 on final list for get_context()
```

## Facet channel (`facets.py`, production port, 2026-09-19 -- OFF by default)
Closed-predicate-vocabulary faceted index, ported from `bench/retrievers.py`'s
`V4PlannedRetriever`/`build_facet_index`/`rank_facet_pool` (measured on LoCoMo only: R@10
0.721 → 0.757 with `facet_weight=2.0`, no reranker -- see `config.py`'s `facet_channel`/
`facet_weight`). A facet is a client-LLM-supplied `(subject, predicate, object)` triple;
`predicate` is one of `facets.PREDICATES` (43 predicates mirrored from
`bench/data/enrich/PREDICATES.md`) or `"other"` (accepted, never indexed). `MemoryItem.facets`
(see "Data model" above) is always populated and indexed by `facets.FacetIndex` -- an in-memory
`(subject_key, predicate) -> {item_id}` map -- regardless of the flag, so enabling it later needs
no re-index; only the READ path is gated. `FacetIndex` is **derived state only**: no on-disk file
of its own, fully rebuilt from `items.jsonl`-loaded `MemoryItem.facets` by
`memory.MemoryV4._reconcile_indexes()` at load, kept in sync incrementally by `add`/`forget`/
`consolidate()`'s dedup-triggered rebuild. `HybridRetriever.facet_pool(facets, kinds, project,
at_time)` normalizes the caller's query facets, looks up the union pool, and filters it through
the exact same `_passes_filters` rule `search()` applies to every ordinary candidate. When
`config.facet_channel` is on and that pool is non-empty, `search()`/`get_context()` route through
`search_multi` (even with no `queries`/`hypotheses`), which fuses `facets.rank_facet_pool(pool,
id_lists)` as one more RRF party (`config.facet_weight`); every facet-pool member (pre-existing
hit or synthesized) is tagged `signals["facet"] = True`. Flag off, no facets, or an empty pool:
byte-identical to today. `instructions.py`'s `build_instructions(config)` appends a facet-channel
paragraph (predicates included, <= 900 chars) only when the flag is on; `MEMORY_INSTRUCTIONS`
itself is unchanged.

## On-disk layout (`MEMORY_V4_DIR`, default `~/.memory-os-v4/`)
```
items.jsonl   embeddings.npy   ids.json   chunk_map.json (multi-vector: chunk→item)   bm25.json   entities.json   blocks.jsonl   meta.json
```
`facets.FacetIndex` adds NO new file here -- see "Facet channel" above.

## Evaluation (`bench/`)
- `download.py`: LoCoMo `locomo10.json` (2.8 MB, GitHub snap-research) ; LongMemEval `longmemeval_oracle` (15 MB, HF xiaowu0162) ; LongMemEval-S (278 MB) optional.
- `locomo.py`: each dialog turn → episode (speaker, text, session date → event_time); question → evidence `dia_id`s.
- `metrics.py`: Recall@k, MRR, nDCG@k per category; macro over categories 1-4.
- `run_bench.py --config name`: builds index from scratch, runs all questions, writes `results/<name>_vN.json` (auto-increment N) and appends a row to `RESULTS.md`.
- Baseline `v3equiv`: 128-char chunks, MiniLM, FlatL2 — implemented in bench only (never imports v3 code).

## Façade signatures (contract shared by core, bench and server)
```python
class MemoryV4:
    def __init__(self, data_dir: str | None = None, config: Config | None = None): ...
    def add_episode(self, text, role="user", project="default", event_time=None, entities=None,
                    tags=None, importance=0.5, item_id=None, source="chat_save", doc_ref=None,
                    keywords=None, facets=None) -> MemoryItem
    def add_episodes_bulk(self, episodes: list[dict]) -> list[MemoryItem]   # same keys as add_episode; one batched encode
    def add_facts(self, facts: list[dict], project="default", parent_ids=None, event_time=None,
                  source="client") -> dict   # {"added": [ids], "deduped": [ids], "superseded": [(old_id, new_id)]}
                  # each fact dict may carry its own "keywords"/"facets", like add_episode's params
    def add_document(self, name: str, text: str, project="default", doc_ref=None) -> dict
    def search(self, query, top_k=10, at_time=None, kinds=None, project=None, rerank=None,
               queries=None, hypotheses=None, facets=None) -> list[SearchHit]
    def get_context(self, query, max_tokens=1500, at_time=None, project=None,
                     queries=None, hypotheses=None, facets=None) -> str
    def session_brief(self, focus_query=None, max_tokens=3000, project=None) -> dict
    def consolidate(self, dry_run=False) -> dict
    def forget(self, item_id: str, reason: str = "") -> bool   # soft-delete + immediate index removal (A5, additive)
    def status(self) -> dict
    def flush(self) -> None          # persist indexes/caches to data_dir
    def migrate_v3(self, log_path: str) -> dict

@dataclass
class SearchHit:
    item: MemoryItem
    score: float
    signals: dict      # {"dense": {"rank","score"}, "bm25": {...}, "entity": {...}, "fused": float, "recency": float, "query_dates": list[float], "temporal_boost": bool, "rerank": float (only when a reranker ran), "multi_query": {"n_queries"} (search_multi only), "facet": True (facet_channel pool members only)}
```
`item_id` lets callers (bench) supply external ids such as LoCoMo `dia_id`; uniqueness enforced.
Fusion method is `config.fusion in {"rrf", "dbsf", "cc"}`; weights in `config.w_dense/w_bm25/w_entity`.
`query_dates` (A7, "Hardening 2026-09-13"): `temporal.parse_dates(query, reference_time=at_time or now)`,
the same value on every hit for a given `search()` call. `temporal_boost` is `config.temporal_query_mode
== "auto"` AND (`query_dates` non-empty OR the query matches a small EN/FR temporal-cue regex) -- when
true, `w_bm25` is multiplied by `config.temporal_bm25_boost` and recency is disabled, for that query only
(see "Part B" in docs/ABLATIONS.md).

## Bench retriever protocol (`bench/protocol.py`)
```python
@dataclass
class BenchItem: id: str; text: str; speaker: str; event_time: float | None; session_id: str; meta: dict
class Retriever(Protocol):
    name: str
    def reset(self) -> None: ...
    def add(self, items: list[BenchItem]) -> None: ...
    def search(self, query: str, top_k: int, at_time: float | None = None) -> list[str]: ...  # ranked item ids
```
Adapters: `V3EquivRetriever` (128-char chunks, MiniLM, FlatL2, chunk→turn dedup) and `V4Retriever(config_overrides)`.

## MCP server v4 (`server.py`) — tool surface
Compat (same names as v3, same required params, richer output): `memory_session_brief`, `memory_chat_save`, `memory_search`, `memory_get_context`, `memory_status`, `memory_ingest`, `memory_list_documents`, `memory_search_occurrences`, `memory_compact` (alias of consolidate).
New: `memory_add_facts`, `memory_timeline`, `memory_block_get`, `memory_block_set`, `memory_consolidate`, `memory_forget`, `memory_migrate_v3`.

| tool | input (pydantic) | output |
|---|---|---|
| memory_session_brief | focus_query?, max_tokens=3000, project? | `session_brief()` dict (blocks first) |
| memory_chat_save | messages[{role,content,keywords?,facets?}], summary?, project_label="default", facts?[FactIn], event_time? (ISO) | {episodes: n, summary_id?, facts: {added, deduped, superseded}} |
| memory_add_facts | facts[FactIn], project?, parent_ids? | same facts dict |
| memory_search | query, top_k=10, at_time? (ISO), kinds?, project?, rerank?, queries?, hypotheses?, facets?[{subject,predicate}] | hits[{id, kind, text, score, event_time, valid_from, valid_until, entities, project, signals}] |
| memory_get_context | query, max_tokens=1500, at_time?, project?, queries?, hypotheses?, facets?[{subject,predicate}] | {context: str, n_items} |
| memory_timeline | query (entity or topic), project?, limit=30 | facts sorted by event_time with validity + invalidated_by |
| memory_block_get | name? | {blocks: {name: text}} or one |
| memory_block_set | name, text, mode="replace"\|"append" | {name, chars, truncated: bool} |
| memory_consolidate | dry_run=false | counts + summary_proposals |
| memory_forget | item_id? or query?, reason | {forgotten: [ids]} (soft delete only) |
| memory_ingest | path (file or folder), project?, extensions? | {documents, chunks} (txt/md/json/py…; pdf via pymupdf if importable) |
| memory_migrate_v3 | log_path? (default ~/.memory-os-ai/_conversation_log.jsonl) | counts |
`FactIn = {text: str, kind?: "fact"|"summary"|"procedure"|"belief", entities?: [str], event_time?: ISO, valid_from?: ISO, valid_until?: ISO, importance?: 0-1, tags?: [str], confidence?: 0-1, keywords?: [str], facets?: [{subject, predicate, object?}]}`
`facets`/query-`facets` (facets.py, OFF by default via `config.facet_channel`): `predicate` should be one of `facets.PREDICATES` (else normalized to `"other"`, stored but never indexed); query-side facets carry no `object`. See "Facet channel" above.
Transport: stdio (default) via `mcp.server.Server`; entry `python -m memory_os_v4.server`. Env: MEMORY_V4_DIR, MEMORY_V4_EMBED_MODEL, MEMORY_V4_PROJECT (default project label), MEMORY_V4_RERANK_MODEL, MEMORY_V4_LLM_URL/MODEL. Path traversal blocked for ingest/migrate (must resolve under $HOME or MEMORY_WORKSPACE).

## MCP instructions (`instructions.py`) — client-side extraction protocol
1. Start of session: call `memory_session_brief` before answering (blocks + recent + key facts + procedures + pending).
2. After significant work, every ~10 exchanges, and before ending: call `memory_chat_save` with messages, a memo-style `summary`, AND `facts`: atomic, self-contained statements (one fact = one sentence, subject explicit, no pronouns), ADD-only (never ask the server to delete; if something changed, state the new fact with its date and `replaces` ids or an explicit single-valued `identity` — similarity alone never supersedes), include: user preferences/constraints, decisions with rationale, results with numbers, file/project locations, tasks (tag "todo"), lessons from successes AND failures (kind="procedure", title-like first clause), agent actions as facts ("Claude created ~/x/y.py on 2026-09-12"). Add `entities` (names, projects, files, tickers) and `event_time` when known.
3. Use `memory_search` with `at_time` for "what was true on <date>" questions and `memory_timeline` for "how did X evolve".
4. Update core blocks (`memory_block_set`) when the user profile, project state or working rules change; keep each block under its limit; replace, don't accumulate.
5. Run `memory_consolidate` when the brief reports > 2000 items or weekly; write the proposed summaries back via `memory_add_facts(kind="summary")`.
6. Facet channel (only when `config.facet_channel` is on -- `build_instructions()` appends this paragraph, otherwise `MEMORY_INSTRUCTIONS` is unchanged): at save, attach `facets` (subject, predicate from the closed vocabulary, object) to messages/facts; at search, pass `facets` (subject, predicate) for what the question is about.
Prompt wording to reuse in instructions (borrowed from Mem0 v3 / Graphiti / LangMem, verified 2026-09-12):
- Ground every relative time expression against the explicit observation date of the message ("yesterday" → absolute date).
- Entities must be specific: "Nisha's dad", never "dad"; never generalize ("Ford Mustang" must not become "car"; "NQ" must not become "index future").
- Do not mark facts with key differences as duplicates ("software engineer" vs "senior engineer" is an update, not a duplicate).
- Prefer dense, complete facts over overlapping ones; caveat uncertain information with a confidence value.
