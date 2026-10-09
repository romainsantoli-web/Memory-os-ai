# Memory OS v4 — evaluation bench

Model-free retrieval evaluation (Recall@k / MRR / nDCG@k against gold evidence
ids — no LLM judge) on LoCoMo and LongMemEval-oracle, comparing the old v3
chunking scheme, dense/BM25 baselines, and the v4 core once it lands. See
`../PLAN.md` §5 (phase P2) and `../docs/ARCHITECTURE.md` ("Evaluation" /
"Bench retriever protocol" sections) for the design contract this package
implements.

## Quickstart

```bash
cd /Users/romain/Memory-os-ai/v4

# 1. download the two datasets into bench/data/ (skips if already present)
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.download --dataset all

# 2. run a retriever on a dataset
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v3equiv --name locomo_v3equiv --top-k 10

# smoke-test on a handful of conversations/questions first
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever bm25 --name smoke --limit 2 --no-ledger

# v4 core (once memory_os_v4.MemoryV4/Config exist), with config overrides
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4 --name locomo_v4_rrf \
    --config-json '{"fusion":"rrf","use_bm25":true}'
```

Every run writes `bench/results/<name>_v<N>.json` (`N` auto-increments —
nothing is ever overwritten) and appends one row to the project-level
`../RESULTS.md` ledger (unless `--no-ledger`).

## Datasets

- **LoCoMo10** (`bench/data/locomo10.json`, ~2.7 MB, snap-research/locomo,
  arXiv:2402.17753): 10 long conversations (avg. ~590 turns / ~19 sessions
  each), 2 825 QA pairs across 5 categories. `bench/locomo.py`'s docstring
  documents the exact JSON shape and the empirically-verified category
  mapping:

  | category | name | n | avg #evidence | notes |
  |---|---|---|---|---|
  | 1 | multi-hop | 282 | 3.13 | 97.9% of questions need >=2 evidence turns |
  | 2 | temporal | 321 | 1.17 | uniformly "When did ... ?" |
  | 3 | open-domain | 96 | 2.08 | counterfactual/inferential ("Would X still...") |
  | 4 | single-hop | 841 | 1.07 | one direct quoted fact |
  | 5 | adversarial | 446 | 1.03 | `answer` is always null; excluded (no ground truth to retrieve for) |

  4 more questions (all category 3) have an empty `evidence` list and are
  also excluded. **Caveat**: independent audits (Penfield Labs, see
  `../docs/RESEARCH_NOTES.md`) found ~6.4% of the LoCoMo answer key is wrong;
  we additionally found 9/2815 (0.3%) evidence ids that reference dia_ids
  absent from their own conversation (can never be recalled — left as-is,
  since silently dropping them would inflate scores). Treat deltas smaller
  than a few points as noise.

- **LongMemEval oracle** (`bench/data/longmemeval_oracle.json`, ~14.7 MB,
  HF `xiaowu0162/longmemeval`, ICLR 2025 arXiv:2410.10813): 500 questions,
  6 question types. **Oracle variant caveat**: verified empirically that for
  all 500 questions `haystack_session_ids == answer_session_ids` — i.e. there
  are NO distractor sessions in this file at all (unlike the -s/-m variants),
  so session-level retrieval is trivial by construction; what the bench
  actually measures here is **turn-level** recall (evidence = turns flagged
  `has_answer: true`), which is still sparse: only 896/10 960 turns (8.2%).
  21/500 questions have zero `has_answer` turns in their haystack (mostly
  `multi-session`/`knowledge-update` questions whose answer is a synthesis,
  not a single quoted turn) and are excluded, same rule as LoCoMo's empty
  evidence. Because there are no distractors, absolute recall numbers here
  are **not comparable** to LoCoMo's or to published numbers on -s/-m/full
  LongMemEval — use this only to compare retrievers against each other on
  the same file.

Neither `longmemeval_s` (278 MB) nor `longmemeval_m` are downloaded or used
anywhere in this bench (disk budget).

## Retrievers (`bench/retrievers.py`)

- `v3equiv` (`V3EquivRetriever`): reproduces the OLD v3.2.0 engine's
  indexing scheme from scratch (never imports v3): 128-character sliding
  windows, 28-char overlap (100-char step), `all-MiniLM-L6-v2`
  (normalized), FAISS `IndexFlatL2`. Search fetches `top_k*8` chunks, maps
  chunk→parent item id, dedupes preserving first-occurrence order, cuts to
  `top_k`. v3 had no temporal logic at all, so `at_time` is accepted but
  ignored — faithful, not a bug.
- `dense` (`DenseTurnRetriever`): same embedding model, whole turn as one
  vector, FAISS `IndexFlatIP`. Intermediate baseline between v3equiv and
  hybrid retrieval.
- `bm25` (`BM25TurnRetriever`): self-contained Okapi BM25 (k1=1.2, b=0.75,
  inverted-index based), whole turn as one document. No dependency on
  `memory_os_v4` or any external BM25 package.
- `v4` (`V4Retriever`): thin adapter over `memory_os_v4.MemoryV4` (façade
  documented in `../docs/ARCHITECTURE.md`). Importing `bench.retrievers`
  always succeeds; only *instantiating* `V4Retriever` raises `ImportError`
  if the core package doesn't expose `MemoryV4`/`Config` yet.
- `v4_enriched` (`V4EnrichedRetriever`): `V4Retriever` plus offline LLM
  enrichment — see "Enriched variant" below.

The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is loaded at
most once per process (module-level cache in `bench/retrievers.py`) and
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are set before it is ever
imported — no other model is ever downloaded or loaded by this bench.

## Enriched variant (`v4_enriched`)

**What it measures.** A-Mem/Mem0-style memory systems don't index raw
conversation turns verbatim — an LLM first *distills* each turn into
atomic facts (plus keywords/entities), and those distilled facts are what
gets embedded/indexed. `V4EnrichedRetriever` (`bench/retrievers.py`) tests
whether that same trick helps `memory_os_v4`'s own hybrid retrieval, using
enrichment produced **offline**, once, by Sonnet 5 sub-agents reading each
LoCoMo turn (never at query time, never inside the retriever itself) and
written to `bench/data/enrich/locomo_facts/<conv_id>.jsonl` — one JSON line
per turn: `{"turn_id", "facts": [...], "keywords": [...], "entities": [...]}`
(loaded by `bench/enrich.py::load_enrichment`, tolerant of a file being
partially written or entirely absent).

For every turn with extracted facts, `V4EnrichedRetriever.add()` indexes the
raw turn exactly like `V4Retriever` does (same item id), then calls
`MemoryV4.add_facts(...)` once per turn with `parent_ids=[turn_id]` so each
fact is a real `kind="fact"` item pointing back at the turn it came from.
Two modes (`--enrich-mode`):

- `facts` (default): only the extracted facts are added as new `fact` items;
  the raw turn's own indexed text is untouched.
- `facts+keywords`: same, plus `" | keywords: k1, k2, ..."` is appended to
  the raw turn's own text *before* it is embedded/BM25-indexed, so lexical/
  dense signals over the raw turn see the extracted keywords too.

`search()` asks `memory_os_v4` for `top_k*3` candidate hits (mixing
`episode` and `fact` kinds), maps every `fact` hit back to its parent
turn id(s) (an `episode` hit maps to itself), dedupes preserving rank
order, and returns the first `top_k` turn ids — so the bench still scores
against the same turn-id gold evidence as every other retriever. How often
the *top* raw hit was a fact vs. an episode is tracked in
`retriever.search_stats` and reported in the run's `metadata.search_stats`.

**Running it:**

```bash
cd /Users/romain/Memory-os-ai/v4
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_enriched --name locomo_v4_enriched \
    --enrich-mode facts \
    [--enrich-dir bench/data/enrich/locomo_facts]   # this is the default
```

Because the enrichment files are produced by separate, concurrently-running
workers, `run_bench.py` **fails fast** (before indexing or searching
anything) if `--enrich-dir` is missing a file for any conversation the run
needs — pass `--allow-partial-enrich` to run anyway; coverage (turns with
facts / total turns, facts added, conversations missing a file) is then
recorded in the results JSON's `metadata.enrichment` and summarized in the
RESULTS.md ledger row's Notes column either way. `bench/enrich.py` also
exposes a standalone `python -m bench.enrich [--enrich-dir ...]
[--turns-dir ...]` coverage printer for checking enrichment job progress
without running the bench at all.

**Caveat.** This variant exists to test a *mechanism* (does LLM-distilled
fact indexing improve `memory_os_v4`'s hybrid retrieval on LoCoMo?), not to
ship a production feature: the enrichment is generated once, offline, by a
separate LLM call per turn outside `memory_os_v4` entirely, so a real
deployment would need to decide when/how to run that extraction online
(and pay its latency/cost) — nothing in `src/memory_os_v4` changed to
support it. Also see the cost note on `V4EnrichedRetriever` in
`bench/retrievers.py`: `MemoryV4.add_facts` runs dedup/near-duplicate/
supersession scans over the whole store per fact, so indexing time grows
faster than linearly with facts-per-conversation — fine at LoCoMo's scale,
not something to extrapolate to much larger conversations unchanged.

## Planned variant (`v4_planned`) — P9

**What it measures.** LoCoMo's remaining weak categories after the
enrichment campaign (`docs/ABLATIONS.md` "Enrichissement par LLM hors
ligne") are multi-hop (evidence spread across ≥3 turns) and open-domain
(inference beyond any single quoted sentence) — a single retrieval query
can't easily surface scattered evidence written in different vocabulary
than the question. `V4PlannedRetriever` (`bench/retrievers.py`, extends
`V4EnrichedRetriever`) tests three P9 (`PLAN.md` §P9) mechanisms on top of
the enriched turns+facts index:

- **§A query planning**: each question is rewritten **offline**, once, by
  Sonnet 5 into 2-4 subqueries (in the dialogue's own vocabulary) and 1-2
  hypothetical facts (HyDE-style), written to
  `bench/data/enrich/locomo_queries/<conv_id>.jsonl` — one JSON line per
  question: `{"qid", "subqueries": [...], "hypotheses": [...]}` (loaded by
  `bench/enrich.py::load_query_plans`, same missing/malformed/duplicate
  tolerance contract as `load_enrichment`). `search()` runs
  `MemoryV4.search(..., rerank=False)` once per component (the original
  question plus every selected subquery/hypothesis, `top_k*4` oversample
  each), maps every hit list to turn ids exactly like `V4EnrichedRetriever`
  does, and fuses the resulting ranked lists with weighted Reciprocal Rank
  Fusion (`bench.retrievers.rrf_fuse`, `k=60` by default;
  `--plan-weight-original/-sub/-hyp` control each component's weight, default
  `1.0/1.0/0.7`). `--plan-mode {sub, sub+hyp, hyp}` selects which components
  beyond the original question are used. A question whose `qid` has no plan
  yet (concurrent worker not finished, or `--queries-dir` omitted) falls
  back to searching the original question alone — counted in
  `metadata.planning.n_fallback_missing_plan`, never an error.
- **§B fact links**: an A-Mem/Zettelkasten-style link graph built once per
  conversation over its enriched facts (`V4PlannedRetriever._build_link_graph`):
  two facts are linked if they share ≥1 "rare" entity (document frequency
  over turns ≤5%, ≥3 chars, not a speaker name) OR belong to the same
  session and are within 2 turns of each other. `--fact-links` turns this
  on: after RRF fusion, for the top `--link-seed` (default 8) fused turns,
  every turn owning a fact linked to one of that seed turn's own facts is
  added at `seed_score * --link-decay` (default 0.5), before reranking.
- **§C reranker**: `--rerank-model` (e.g.
  `tomaarsen/ms-marco-ettin-17m-reranker` or the larger `-68m` variant,
  downloaded once via `huggingface_hub.snapshot_download`, `HF_HUB_OFFLINE=1`
  for every run afterwards) is applied by the **bench itself**, once, to the
  top `--rerank-top-n` (default 30) fused (and, if `--fact-links`,
  link-expanded) turns — deliberately independent of
  `memory_os_v4.Config.rerank_model` (every internal `MemoryV4.search()`
  call above passes `rerank=False`; see `V4PlannedRetriever`'s class
  docstring). Scores `(question, turn_text [+ its facts joined])` pairs and
  either blends 0.5/0.5 with the pre-rerank fused score (`--plan-rerank-blend`,
  default, mirrors `memory_os_v4.retrieval`'s own blend) or ranks by the
  cross-encoder score alone (`--plan-rerank-ce-only`).

**Running it:**

```bash
cd /Users/romain/Memory-os-ai/v4
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_planned --name locomo_v4_plan_subhyp \
    --enrich-mode facts+keywords --plan-mode sub+hyp \
    --rerank-model tomaarsen/ms-marco-ettin-17m-reranker --rerank-top-n 30 \
    [--queries-dir bench/data/enrich/locomo_queries]   # this is the default
    # optional: --fact-links [--link-seed 8] [--link-decay 0.5]
    # optional: --plan-rerank-ce-only (default is --plan-rerank-blend)
    # optional: --rrf-k 60 [--plan-weight-original 1.0] [--plan-weight-sub 1.0] [--plan-weight-hyp 0.7]
```

Because the query-plan files are produced by separate, concurrently-running
workers (same as `locomo_facts`), `run_bench.py` **fails fast** if
`--queries-dir` is missing a file for any conversation the run needs — pass
`--allow-partial-plan` to run anyway (every question in a missing
conversation falls back to single-query search); coverage
(`n_plan_used`/`n_fallback_missing_plan`/`fraction_questions_with_plan`) is
recorded in `metadata.planning` and summarized in the RESULTS.md ledger
row's Notes column either way. `V4PlannedRetriever` still needs
`--enrich-dir`/`--enrich-mode` (it extends `V4EnrichedRetriever`) — the same
fail-fast/`--allow-partial-enrich` rule applies to that dir independently.

**Caveat.** Like the enriched variant, this tests a *mechanism* end to end
offline (query rewriting is a Sonnet 5 call outside `memory_os_v4`, run
once per question, not at serve time) — nothing in `src/memory_os_v4`
changed to support it. Latency also compounds: up to 5 internal searches
per question (no rerank each) plus one bench-side cross-encoder pass over
the fused top-n makes `v4_planned` meaningfully slower per query than
`v4_enriched` alone (see `docs/ABLATIONS.md` "P9" for measured p50/p95).

### Idées inter-disciplines (`docs/IDEAS.md`, 2026-09-14)

Five more `V4PlannedRetriever` options, all **off by default**, attacking
*coverage* (does the right turn even reach the fused top-K?) rather than
similarity, per `docs/IDEAS.md`'s diagnosis that multi-hop/open-domain are
capped by which candidates survive fusion, not by how well they're scored.
Applied in this order after §A fusion and before §C rerank: idea 1
(`--fusion-mode`) replaces the fused ORDER; idea 5 (`--iterative-budget`)
fills starved sub-query coverage gaps; §B fact-links (existing) and idea 4
(`--adjacency-pair`) both ADD candidates; idea 2 (`--subquery-mmr`)
re-orders the result one last time. Idea 3 (`--signal-confidence`) instead
changes §A's own fusion WEIGHTS. See `docs/ABLATIONS.md` "Idées
inter-disciplines (2026-09-14)" for measured results.

- **Idea 1 — `--fusion-mode {rrf, quota, quota_then_rrf}`** (Sainte-Laguë
  proportional fusion, social-choice theory): plain RRF (default) lets the
  "loudest" query list sweep the fused top-K. `quota` instead treats every
  query list (original question, each sub-query, each hypothesis) as a
  party with a fixed weight (`--plan-weight-original/-sub/-hyp`) and
  allocates the first `--rerank-top-n` positions one seat at a time by the
  Sainte-Laguë divisor method (`bench.retrievers.quota_fuse`) — every
  non-empty party gets at least one of the first few seats, then anything
  left over is appended in plain RRF order. `quota_then_rrf` allocates only
  the first `--quota-then-rrf-boundary` (default 10) positions by quota,
  plain RRF after.
- **Idea 2 — `--subquery-mmr` [`--mmr-lambda 0.6`]** (Markowitz portfolio
  diversification): after fusion (and any idea-1/-4/-5/§B expansion), a
  greedy re-order (`bench.retrievers.subquery_mmr_reorder`) penalises a
  candidate by `mmr_lambda ** n` once its "owner" query list (the list
  where it ranks best, `_compute_owner_map`) is already represented `n`
  times among already-selected candidates — ties broken by the
  unpenalised fused score. Runs last, right before §C rerank.
- **Idea 3 — `--signal-confidence`** (sensor fusion / Kalman gain): scales
  each query list's RRF weight by `bench.retrievers.signal_confidence` —
  the normalised margin between that list's own top-1 and top-2 RAW ENGINE
  scores (before turn-id mapping), clipped to `[0.3, 1.0]`. A "flat" list
  (top-1 barely beats top-2) is down-weighted toward the floor; a list is
  never boosted above its configured base weight.
- **Idea 4 — `--adjacency-pair`** (discourse linguistics, adjacency pairs):
  for each of the top-10 fused turns whose raw text reads as a question
  (`bench.retrievers._is_question_turn` — ends with "?" or opens with a
  5W1H/yes-no question word) AND whose speaker differs from the speaker of
  the immediately-following SAME-SESSION turn (this conversation's own
  transcript order, recorded by `add()` — never re-derived from the v4
  store), that following turn is added at `seed_score * 0.8` — never
  overriding a turn already present (same "only ever ADD" convention as
  §B).
- **Idea 5 — `--iterative-budget`** (optimal foraging theory, Pirolli):
  after the first fusion, any sub-query with NONE of its own hits in the
  fused top-20 is re-run alone at `top_k * 8` and its own top-5 merged into
  fused positions 11-20 (ahead of whatever already occupied them) — capped
  at 2 extra `MemoryV4.search()` calls per question (further-starved
  sub-queries beyond the cap are simply left starved).

**Recall ceilings — `--report-ceilings`.** Independent of the ideas above:
computes recall@k for k in (10, 20, 30, 50) of the FINAL ranked list from a
single `search(top_k=50)` call per question (the normal `--top-k` metrics
are computed from the SAME call's `[:top_k]` prefix, so this never doubles
latency) — evidence is used only to SCORE the already-computed list
afterwards, never fed back into retrieval. Written to
`metadata.ceilings.{per_category,macro,micro}` (keys `r10`/`r20`/`r30`/`r50`)
and printed in the run summary; works with any `--retriever`, not just
`v4_planned`.

```bash
cd /Users/romain/Memory-os-ai/v4
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_planned --name locomo_v4_idea1_quota \
    --enrich-mode facts+keywords --plan-mode sub+hyp --rrf-k 20 \
    --rerank-model tomaarsen/ms-marco-ettin-68m-reranker --rerank-top-n 30 --top-k 10 \
    --report-ceilings --fusion-mode quota
    # other ideas, one at a time: --fusion-mode quota_then_rrf [--quota-then-rrf-boundary 10]
    #   | --subquery-mmr [--mmr-lambda 0.6] | --signal-confidence | --adjacency-pair | --iterative-budget
```

### Facet channel + implications (`docs/IDEAS.md` idea 8, 2026-09-19)

Two more `V4PlannedRetriever` options, both **off by default** and
independent of each other and of the idées inter-disciplines above. Idea 8's
size-matched offline test (`bench/facet_size_matched.py`) already showed
that a faceted index with a closed predicate vocabulary raises the recall
CEILING for multi-hop questions beyond what the same number of ordinary
candidates gives (+0.023 macro at 50+k candidates, positive in 10/10
conversations) — these options test whether that ceiling gain survives
being ranked into the actual top 10. See `docs/ABLATIONS.md` "V5.0 — canaux
facettes et implications (2026-09-19)" for measured results.

- **`--facet-channel` [`--facet-weight 0.5`]**: at `add()` time, builds a
  per-conversation index `(subject, predicate) -> {turn_id, ...}` from
  `--facets-dir` (default `bench/data/enrich/locomo_facets/<conv_id>.jsonl`
  — one line per extracted fact, `{"fid", "facets": [{"subject",
  "predicate", "object"}], "implications"}`), skipping predicate `"other"`
  (`bench.retrievers.build_facet_index`). At `search()` time, looks up the
  question's own facets by `qid` from `--qfacets-dir` (default
  `bench/data/enrich/locomo_qfacets/<conv_id>.jsonl` — `{"qid", "facets":
  [{"subject", "predicate"}]}`), takes the union of matching turns (the
  "facet pool"), and turns it into a ranked list (`bench.retrievers.
  rank_facet_pool`: ordered by each member's own best rank across the
  original/sub-query/hypothesis hit lists already computed for this
  question; members in none of them go last, ordered by turn id for
  determinism) added to the RRF fusion as one more party at `--facet-weight`
  — BEFORE `rrf_fuse` runs, so `--rerank-top-n` and every downstream idea
  see facet-injected candidates like any other fused candidate. Exact
  predicate match only (predicate-family widening was tested offline and
  showed no advantage over exact, see `docs/IDEAS.md`). A question with no
  facets, a conversation with no facet file, or `--facet-weight 0` leaves
  the fusion byte-identical to `--facet-channel` being off.
- **`--implications`**: at `add()` time, indexes each fact's implications
  (same `--facets-dir` file) as additional `add_facts` items — one batch per
  turn, `parent_ids=[turn_id]`, `tags=["implication"]`, `importance=0.5`,
  same mechanism/cost profile as the enriched-facts indexing in
  `V4EnrichedRetriever.add`. A hit on an indexed implication is credited to
  its parent turn by the same fact→parent mapping every other fact-kind hit
  already goes through — no separate credit logic needed.

Both options report coverage in `metadata.facets`: `facet_channel`,
`facet_weight`, `implications`, `n_questions`/`n_questions_with_facets`/
`fraction_questions_with_facets`, `mean_pool_size` and
`mean_new_vs_fused_top50` (mean, over every question the facet channel ran
for, of how many pool members were NOT already in the plain pre-facet fused
top-50 — informational only, never affects ranking), and
`n_implications_indexed`.

```bash
cd /Users/romain/Memory-os-ai/v4
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_planned --name locomo_v5_facets_w05 \
    --enrich-mode facts+keywords --plan-mode sub+hyp --rrf-k 20 --top-k 10 \
    --facet-channel --facet-weight 0.5
    # other configs: --facet-weight 1.0 | --implications | --facet-channel --implications
    #   | add --rerank-model tomaarsen/ms-marco-ettin-68m-reranker --rerank-top-n 50 for the best config
```

## LLM listwise reranking, offline (`v4_llmrank`) — P9 step D

**What it measures.** `v4_planned`'s remaining gap to A-Mem (`docs/ABLATIONS.md`
"P9") is concentrated in multi-hop/open-domain questions where the right
turn is IN the top-20 (a "ranking" problem, not a "recall" problem) but not
at rank 1-10 (a "cross-encoder only sees one pair at a time" problem, per
P9's own §A/§C interpretation notes). P9 step D tests whether an LLM,
shown all 20 candidates AT ONCE (listwise, not pairwise), can reorder them
better than the bench's own cross-encoder — entirely **offline**, in two
stages that never touch `src/memory_os_v4`:

1. **`--dump-candidates DIR`** (any retriever, typically `v4_planned` with
   the best-so-far config): writes `DIR/<conv_id>.jsonl`, one line per
   question, recording the retriever's FINAL top-20 ranking (turn id, date,
   speaker, text truncated to 320 chars, up to 2 of its own enriched facts
   truncated to 160 chars each, and a `"c01".."c20"` label) plus a compact
   `label_map` — see `bench.enrich.build_candidate_dump_line`. **Never**
   includes evidence ids, answers, or categories (an LLM reranking worker
   must not see the gold labels it's meant to help predict). A single
   oversampled `search(top_k=max(--top-k, 20))` call serves both the dump
   AND this run's own normal `--top-k` metrics (`ranked[:top_k]` is an exact
   prefix of the dumped top-20) — dumping never doubles latency.
2. An LLM (any worker, any session, run later, completely outside this
   bench) reads each dumped question + its 20 labeled candidates and writes
   `<llmrank_dir>/<conv_id>.jsonl`: `{"qid", "ranked": ["c07", "c02", ...]}`
   — its own best-first ranking of (some or all of) those labels.
3. **`V4LLMRerankedRetriever`** (`--retriever v4_llmrank`, extends
   `V4PlannedRetriever`) applies that ranking: `search()` runs the full
   parent pipeline to get this question's own top-20 (same deterministic
   config as the dump), translates the LLM's labels to turn ids via the
   dump's own `label_map` (`bench.enrich.load_candidate_dump`) — **not** by
   trusting a freshly-recomputed list's Nth position — via
   `bench.retrievers.apply_llm_ranking`: the LLM-ranked ids first (deduped),
   then the remaining top-20 candidates in their original order, cut to
   `top_k`. A question with no LLM ranking yet (or every one of its labels
   failed to map) falls back to plain `v4_planned` order — counted in
   `metadata.llm_rerank.n_fallback_no_llm_ranking`; `llm_ranked_questions`
   counts how many searches actually had a usable ranking applied.

**Running it:**

```bash
# 1) dump candidates (uses the best-so-far v4_planned config, P9 "Planned variant" above)
cd /Users/romain/Memory-os-ai/v4
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_planned --name locomo_v4_plan_best_k20_dump --top-k 10 \
    --enrich-mode facts+keywords --plan-mode sub+hyp --rrf-k 20 \
    --rerank-model tomaarsen/ms-marco-ettin-68m-reranker --rerank-top-n 30 --plan-rerank-blend \
    --dump-candidates bench/data/dumps/locomo_v4_plan_best_k20

# 2) (elsewhere, offline, later) an LLM worker reads the dump and writes
#    bench/data/dumps/locomo_v4_plan_best_k20_llmrank/<conv_id>.jsonl
#    {"qid", "ranked": ["c07", "c02", ...]} lines -- not part of this bench.

# 3a) evaluate by recomputing retrieval (slow, exercises the real engine):
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_llmrank --name locomo_v4_llmrank --top-k 10 \
    --enrich-mode facts+keywords --plan-mode sub+hyp --rrf-k 20 \
    --rerank-model tomaarsen/ms-marco-ettin-68m-reranker --rerank-top-n 30 --plan-rerank-blend \
    --candidates-dir bench/data/dumps/locomo_v4_plan_best_k20 \
    --llmrank-dir bench/data/dumps/locomo_v4_plan_best_k20_llmrank

# 3b) OR evaluate purely from the two dumps (fast, no engine/model at all,
#     latency_ms reported as 0.0 -- see metadata.from_dump.note):
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_llmrank --from-dump --name locomo_v4_llmrank_fromdump --top-k 10 \
    --candidates-dir bench/data/dumps/locomo_v4_plan_best_k20 \
    --llmrank-dir bench/data/dumps/locomo_v4_plan_best_k20_llmrank
```

`--from-dump` (step 3b) never builds a retriever, loads a model, or calls
`add()`/`search()` — it reads `--candidates-dir` (+ optional `--llmrank-dir`)
straight off disk and scores with the same `bench.metrics` functions the
live loop uses, via the exact same `bench.retrievers.apply_llm_ranking`
function `V4LLMRerankedRetriever.search` calls (so the two modes can never
disagree on how a ranking is combined with the dump order). Running 3b with
no `--llmrank-dir` at all reproduces the dump's own ceiling (its top-`top_k`
metrics) — a useful sanity check before any LLM ranking exists yet.
`--retriever` is ignored in `--from-dump` mode (`--candidates-dir` is the
only required flag); the written result's `"retriever"` is always
`"v4_llmrank_from_dump"`.

## Metrics (`bench/metrics.py`)

- `recall_at_k(ranked, evidence, k)`: `|top_k ∩ evidence| / |evidence|`.
- `mrr(ranked, evidence)`: `1 / rank` of the first relevant id (0 if none in
  the full ranked list).
- `ndcg_at_k(ranked, evidence, k)`: binary-relevance nDCG (`rel ∈ {0,1}`),
  normalized by the ideal DCG for `min(|evidence|, k)` relevant items.
- `aggregate(rows)`: per-category means, **macro** (mean of the per-category
  means — the pilot metric used for the RESULTS.md ledger, since it does not
  let the 841-question single-hop category dominate the 96-question
  open-domain one), **micro** (mean over all questions, unweighted by
  category), plus latency p50/p95/mean and mean returned-context size
  (`chars/4` of the top-10 texts, a cheap token proxy — no tokenizer dep).

All three metric functions treat an empty evidence list as score 0.0 rather
than raising, but `run_bench.py`/the loaders are expected to filter those
questions out before scoring (LoCoMo category 5 + empty evidence; LongMemEval
questions with zero `has_answer` turns) since there is no ground truth to
score against.

## Adding a new run

```bash
PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset <locomo|longmemeval_oracle> --retriever <v3equiv|dense|bm25|v4|v4_enriched|v4_planned|v4_llmrank> \
    --name <descriptive_name> [--limit N] [--top-k 10] [--config-json '{...}']
    # v4_enriched/v4_planned/v4_llmrank: [--enrich-dir DIR] [--enrich-mode facts|facts+keywords] [--allow-partial-enrich]
    # v4_planned/v4_llmrank (P9, see "Planned variant" above): [--queries-dir DIR] [--plan-mode sub|sub+hyp|hyp]
    #   [--rerank-model NAME] [--rerank-top-n 30] [--plan-rerank-blend|--plan-rerank-ce-only]
    #   [--fact-links] [--link-seed 8] [--link-decay 0.5] [--rrf-k 60] [--allow-partial-plan]
    # v4_planned only (idées inter-disciplines, see above): [--fusion-mode rrf|quota|quota_then_rrf]
    #   [--quota-then-rrf-boundary 10] [--subquery-mmr [--mmr-lambda 0.6]] [--signal-confidence]
    #   [--adjacency-pair] [--iterative-budget]
    # v4_planned only (idea 8, facet channel + implications, see above): [--facet-channel [--facet-weight 0.5]]
    #   [--facets-dir DIR] [--qfacets-dir DIR] [--implications]
    # any retriever: [--report-ceilings]  (recall@10/20/30/50 of the final ranked list, see above)
    # any retriever (P9 step D, see "LLM listwise reranking" above): [--dump-candidates DIR]
    # v4_llmrank only (P9 step D): [--llmrank-dir DIR] [--candidates-dir DIR]
    #   [--from-dump]  (skips the engine entirely; --candidates-dir required, --retriever ignored)
```

`--name` should be descriptive and stable across re-runs of the *same*
approach (e.g. `locomo_v3equiv`); re-running with the same `--name` creates
`_v2.json`, `_v3.json`, ... and one new ledger row each time — old results
are never overwritten or deleted (competition-mode bookkeeping, see
`~/.claude/CLAUDE.md`).

## Tests

`tests/test_bench_*.py` cover: metrics (hand-computed examples), the LoCoMo
and LongMemEval loaders (inline synthetic fixtures — no network/download
needed to run tests), `V3EquivRetriever`'s chunking math with a fake embedder
(no model load needed), and an end-to-end `run_bench` pass over a synthetic
2-conversation dataset with the `bm25` retriever, writing into a tmp
results dir / tmp ledger file (the real `RESULTS.md` is never touched by
tests). `tests/test_bench_enrich.py` covers `bench/enrich.py`'s tolerance
contract (missing/malformed/duplicate enrichment lines) and
`V4EnrichedRetriever` (fact-hit -> parent-turn mapping, keywords-mode text
augmentation, coverage/search_stats bookkeeping, an end-to-end
`--retriever v4_enriched` run) using `Config(embed_model="hash")` and
synthetic tmp-dir fixtures — never the real, concurrently-written
`bench/data/enrich/locomo_facts/`. `tests/test_bench_plan.py` covers P9:
`bench.enrich.load_query_plans`'s tolerance contract, RRF fusion math
(`rrf_fuse`/`_rank_by_score`), `V4PlannedRetriever` plan lookup/fallback and
`--plan-mode` selection (spying on the underlying `MemoryV4.search` calls),
the §B fact-link graph (rare-entity/same-session-adjacency construction and
link expansion), the §C reranker's blend vs. cross-encoder-only paths (a
fake reranker, same pattern as `tests/test_core_retrieval.py`'s), and an
end-to-end `--retriever v4_planned` run (incl. fail-fast/`--allow-partial-plan`)
— same `Config(embed_model="hash")` + synthetic tmp-dir fixture approach,
never the real, concurrently-written `bench/data/enrich/locomo_queries/`.
The same file also covers P9 step D (LLM listwise reranking, offline):
`bench.enrich.build_candidate_dump_line`/`load_candidate_dump`/
`load_llm_rankings` field shapes, truncation (320/160 chars, max 2
facts/turn, max 20 candidates) and tolerance contracts (missing file,
malformed lines, duplicate qid, a dump line predating the top-level
`label_map` field), `bench.retrievers.apply_llm_ranking` (dedup/fallback/
`top_k` cut math), `V4LLMRerankedRetriever` (label->turn_id mapping via the
dump's own `label_map`, unknown-label and no-ranking fallback to parent
order, `llm_stats` bookkeeping), `run_bench`'s `--dump-candidates` (file/
line counts, exactly the documented candidate fields, never an evidence
id/answer/category), and `run_bench_from_dump` (ceiling reproduction with
no `--llmrank-dir`, ranking applied when one is given, a missing dump
record scoring 0 rather than raising) — all synthetic tmp-dir fixtures,
`latency_ms == 0.0` asserted explicitly wherever `run_bench_from_dump` is
exercised. The same file also covers the idées inter-disciplines
(`docs/IDEAS.md`, 2026-09-14): hand-computed Sainte-Laguë seat allocation
(`quota_fuse`/`quota_then_rrf_fuse`), the MMR owner map and penalty/tie-break
order (`_compute_owner_map`/`subquery_mmr_reorder`), `signal_confidence`'s
clip-to-[0.3, 1.0] contract, `_is_question_turn` detection (mark vs. question
word vs. speaker-prefix stripping), `_adjacency_pair_expand`/
`_iterative_forage` (never-override, cap-at-2-searches, dedup-against-tail),
and `V4PlannedRetriever.search()`-level integration tests for
`--fusion-mode`/`--signal-confidence`/`--subquery-mmr` with fully-canned
`_memory.search` results (`_FakeSearchHit`) plus `run_bench`'s
`--report-ceilings` metadata shape — same synthetic/`embed_model="hash"`
approach throughout. `tests/test_bench_facets.py` covers idea 8 (facet
channel + implications, 2026-09-19): `bench.enrich.load_fact_facets`/
`load_qfacets`'s tolerance contracts (missing file, malformed lines,
"other"-predicate/blank-subject entries dropped, duplicate fid/qid last
wins), `bench.retrievers.build_facet_index` (per-conversation, `"other"`
skipped) and `rank_facet_pool` (best-rank ordering, absent-everywhere and
tie determinism by turn id), `V4PlannedRetriever` integration tests
(a question with no facets / a conversation with no facet file is
byte-identical to the baseline, `--facet-weight 0` reproduces the baseline
ranking exactly, an implication hit is credited to its parent turn, the
facet pool is injected before fusion so `--rerank-top-n` sees it), and the
`metadata.facets` coverage/implications-count fields — same synthetic
tmp-dir + `embed_model="hash"` approach throughout. Run with:

```bash
cd /Users/romain/Memory-os-ai/v4 && PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m pytest tests/test_bench_*.py -q
```
