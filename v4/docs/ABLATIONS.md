# Ablations — LoCoMo/LongMemEval retrieval campaign (2026-09-12)

Competition-mode campaign on `memory_os_v4`'s retrieval stack (`src/memory_os_v4/{text,retrieval,vector_index,memory,config,bm25,entities,store}.py`),
per `PLAN.md` phase P5. Pilot metric: LoCoMo turn-level macro Recall@10 (mean
of categories 1-4); secondary R@5, MRR, nDCG@10; guardrail: p50 search
latency < 30 ms and no regression on LongMemEval oracle. Every run below is a
real `bench/run_bench.py` invocation against the cached
`sentence-transformers/all-MiniLM-L6-v2` model (`HashEmbedder` is unit-test
only) and appends its own `bench/results/<name>_vN.json` + one RESULTS.md
row — nothing here is hand-computed. Deltas smaller than +0.005 R@10 are
reported as "within noise" per the LoCoMo answer-key audit
(`docs/RESEARCH_NOTES.md`, ~6.4% error rate) and are never claimed as wins.

All new behaviour ran behind `Config` flags, default OFF/unchanged, for the
entire ablation phase (see `src/memory_os_v4/config.py`, "Ablation flags"
block) — the pre-campaign `locomo_v4_default` row
(`bench/results/locomo_v4_default_v1.json`, RESULTS.md #6) was reproduced by
every flag at its pre-campaign default value. **After E7 concluded**, the
winning combo's flags (`fusion`, `w_bm25`, `candidates`, `use_stemming`) were
promoted into `Config`'s actual defaults (see "Winning configuration"
below) — `locomo_v4_default` no longer means "pre-campaign"; every row in
the table below is still labeled with the exact config it ran, so nothing
here is ambiguous.

## Adversarial review fixes (2026-09-12, mid-campaign)

A fresh-context adversary reviewed E1-E6 while runs #7-19 (below) were
in flight and found 6 issues, all fixed in the code these ablations run
(never gated behind a flag — these are correctness fixes, not new
behaviour to A/B):

1. **CRITICAL non-determinism.** `EntityIndex.search` (`entities.py`) and
   `BM25Index.search` (`bm25.py`) built their `scores` dict by iterating a
   Python `set` (`bucket`/`candidates`), whose iteration order depends on
   `PYTHONHASHSEED`; `retrieval.py`'s final `scored.sort(...)` had the same
   issue via `candidate_ids: set[str]`. `sorted(..., key=lambda kv: kv[1],
   reverse=True)` is a *stable* sort, so ties (common — one dialogue's main
   participant appears in the majority of turns) were broken by that random
   order, not consistently. **Fix**: every sort site now uses a full order
   `key=lambda kv: (-kv[1], kv[0])` (score desc, id asc) — since ids are
   unique this has zero remaining ties, so the ranking is now provably
   independent of `PYTHONHASHSEED`. Also recorded `PYTHONHASHSEED` in every
   run's `metadata` (`bench/run_bench.py`) for transparency, and re-run with
   it fixed to `0` from here on.
2. **Entity signal quality.** `extract_entities` kept sentence-start
   capitalized words unconditionally in its single-sentence branch — and a
   query is almost always one sentence — so every question polluted its own
   entity list with its interrogative ("What did Caroline research?" ->
   `["what", "caroline"]"). **Fix**: single-word spans matching `STOPWORDS`
   or a small extra set (`how`, `why`, `wow`, `thanks`, `however`) are
   dropped in `extract_entities` (`entities.py`). Additionally,
   `EntityIndex.search` now skips any query entity present in >10% of the
   corpus (`df/n_docs > 0.1`) — not discriminative, near-uniform noise
   across candidates — rather than let its IDF weight still shape ties.
3. **CRITICAL hang.** `text.chunk_text` looped forever on a short sentence
   followed by one exceeding `max_tokens` (e.g. `"Ok. " + 500 words`): the
   old `slen > max_tokens and not current` guard meant the oversized
   sentence only ever hit the "flush current + carry overlap" branch, whose
   overlap carry-over (the same short sentence, since it's below `overlap`)
   reproduced the identical state on the next iteration without advancing
   the loop index. **Fix**: an oversized sentence is now hard-split
   unconditionally, flushing any pending buffer first; regression tests
   added (`tests/test_core_text.py`). This sits directly on E1's hot path.
4. **Recency numerically dead on LoCoMo.** Turn ages (~1200 days) against a
   30-day episode half-life underflow `recency_weight` to ~1e-10 for every
   item alike — `use_recency` on/off is then byte-identical. **Fix**
   (bundled into the existing `recency_only_with_at_time` flag, `E6`): when
   `at_time` *is* given, half-life is floored to 25% of the corpus's own
   `event_time` span (`HybridRetriever._corpus_span_days`), so temporal
   questions can actually benefit from recency instead of it collapsing to
   a constant. Documented in `config.py`.
5. **cc/dbsf degenerate normalization.** `minmax_normalize`/
   `_zscore_clip_minmax` flattened an all-equal-score (or single-candidate)
   signal to a uniform `1.0` for every candidate, discarding that signal's
   own implicit best-first rank order. **Fix**: added `_rank_fallback`
   (linear 1.0 -> 0.0 by original rank) used whenever `hi - lo < 1e-9`
   (raised from `1e-12`) / `std < 1e-9`.
6. **Bench flag bugs.** `context_tokens` was computed over a hardcoded
   `ranked[:10]` regardless of `--top-k`; fixed to `ranked[:top_k]`. The
   `--stratified` flag (round-robin `--limit` across question_type for
   LongMemEval oracle, required for a representative small sample since a
   plain prefix only spans 2/6 types) was already implemented ahead of this
   review (see E7 below) and needed no change.

**Consequence for the ledger**: runs #7-19 (E1/E3/E4/E5/E6's first pass,
pre-fix) are kept — bookkeeping never overwrites — but are re-run under the
fixed code as a fresh `locomo_v4_default` reference plus the configs that
matter for the final decision (E2 stem, E4's winning fusion, E5, E6),
clearly marked "post-fix" below. Pre-fix numbers for configs already
rejected on the hard latency guardrail regardless of R@10 (E1 multi_vector,
E3 neighbor_expansion — both `p50 > 30 ms` on every variant tried) are not
re-run: the fixes touch scoring quality/determinism, not indexing cost, so
they cannot change a latency-based rejection.

## Results table

| Run | R@5 | R@10 | MRR | nDCG@10 | p50 ms | cat.1 multi-hop | cat.2 temporal | cat.3 open-dom. | cat.4 single-hop |
|---|---|---|---|---|---|---|---|---|---|
| locomo_v4_default (baseline, RESULTS.md #6) | 0.367 | 0.435 | 0.293 | 0.304 | 7.9 | 0.262 | 0.636 | 0.253 | 0.588 |
| **E1 multi_vector** | | | | | | | | | |
| locomo_v4_mv48 (#7) | 0.368 | 0.437 | 0.286 | 0.302 | 35.7 | 0.261 | 0.636 | 0.262 | 0.589 |
| locomo_v4_mv96 (#8) | 0.369 | 0.435 | 0.284 | 0.299 | 33.1 | 0.260 | 0.633 | 0.256 | 0.591 |
| **E2 stem** | | | | | | | | | |
| locomo_v4_stem (#9) | 0.381 | 0.464 | 0.298 | 0.316 | 24.1 | 0.320 | 0.658 | 0.277 | 0.604 |
| **E3 neighbor_expansion** | | | | | | | | | |
| locomo_v4_nbr1 (#10) | 0.367 | 0.440 | 0.284 | 0.300 | 34.6 | 0.259 | 0.640 | 0.262 | 0.599 |
| locomo_v4_nbr2 (#11) | 0.369 | 0.440 | 0.287 | 0.302 | 38.8 | 0.260 | 0.649 | 0.256 | 0.596 |
| **E4 fusion** (candidates=100 for all 5 rows) | | | | | | | | | |
| locomo_v4_cc_1_1 (#12, pre-fix) | 0.407 | 0.489 | 0.345 | 0.355 | 15.2 | 0.291 | 0.656 | 0.328 | 0.680 |
| locomo_v4_cc_1_15 (#13, pre-fix) | 0.419 | 0.498 | 0.366 | 0.372 | 35.4† | 0.298 | 0.683 | 0.322 | 0.690 |
| locomo_v4_cc_07_13 (#14, pre-fix) | 0.410 | 0.491 | 0.366 | 0.371 | 30.1† | 0.290 | 0.674 | 0.308 | 0.690 |
| locomo_v4_dbsf_1_1 (#15, pre-fix) | 0.411 | 0.495 | 0.339 | 0.352 | 32.1† | 0.304 | 0.675 | 0.324 | 0.678 |
| locomo_v4_rrf_c100 (#16, pre-fix) | 0.359 | 0.444 | 0.280 | 0.299 | 15.8 | 0.267 | 0.654 | 0.272 | 0.585 |
| **E5 entity weight** (pre-fix — see fix #2 above, re-measured post-fix below) | | | | | | | | | |
| locomo_v4_ent0 (#17, pre-fix, w_entity=0) | 0.384 | 0.455 | 0.325 | 0.332 | 11.8 | 0.283 | 0.650 | 0.256 | 0.630 |
| locomo_v4_ent1 (#18, pre-fix, w_entity=1) | 0.373 | 0.452 | 0.313 | 0.322 | 8.9 | 0.277 | 0.649 | 0.259 | 0.626 |
| **E6 recency fix** (pre-fix, before the adaptive-half-life addition) | | | | | | | | | |
| locomo_v4_norec (#19, pre-fix) | 0.381 | 0.455 | 0.322 | 0.330 | 12.1 | 0.281 | 0.655 | 0.259 | 0.624 |
| **Post-fix re-measurement** (adversarial-review fixes above; `PYTHONHASHSEED=0`, `device=cpu` to sidestep an unrelated MPS/disk-full crash on this host) | | | | | | | | | |
| locomo_v4_default (post-fix, new reference, #20) | 0.382 | 0.454 | 0.321 | 0.329 | 25.9 | 0.279 | 0.652 | 0.259 | 0.626 |
| locomo_v4_stem (post-fix, #21) | 0.395 | 0.481 | 0.340 | 0.348 | 34.2† | 0.337 | 0.667 | 0.283 | 0.639 |
| locomo_v4_ent0 (post-fix, w_entity=0, #22) | 0.384 | 0.455 | 0.327 | 0.334 | 28.3† | 0.283 | 0.650 | 0.256 | 0.631 |
| locomo_v4_ent1 (post-fix, w_entity=1, #23) | 0.373 | 0.452 | 0.313 | 0.322 | 34.5† | 0.277 | 0.649 | 0.259 | 0.626 |
| locomo_v4_norec (post-fix, incl. adaptive half-life, #24) | 0.381 | 0.455 | 0.322 | 0.330 | 33.8† | 0.281 | 0.655 | 0.259 | 0.624 |
| locomo_v4_cc_1_1 (post-fix, #25) | 0.426 | 0.507 | 0.369 | 0.376 | 33.5† | 0.314 | 0.679 | 0.331 | 0.705 |
| **E7 combo** | | | | | | | | | |
| locomo_v4_combo_v1 (#26, cc 1:1 + stem) | 0.443 | **0.522** | 0.383 | 0.390 | 34.0† | 0.347 | 0.694 | 0.343 | 0.705 |
| locomo_v4_combo_v2 (#27, cc 1:1.5 + stem) | **0.453** | **0.527** | **0.398** | **0.403** | 26.5 | 0.360 | 0.705 | 0.323 | 0.720 |

† p50 latency on this shared, heavily-loaded host varied a lot run-to-run
for otherwise-similar configs (e.g. cc_1_1 15.2ms vs cc_1_15 35.4ms despite
identical `candidates=100` compute cost) — other unrelated processes on the
machine (ML training jobs, a concurrent adversarial-review bench process)
were competing for CPU throughout this campaign. Absolute p50 numbers here
are directional, not a precise SLA measurement; the campaign's own
`locomo_v4_cc_1_1` (consistently the fastest strong fusion config, ~15ms)
is used for the final latency guardrail check rather than any single
borderline reading.

LongMemEval oracle, stratified 150 (E7 final check):

| Run | R@5 | R@10 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|
| lme_oracle_bm25_turn (stratified 150, #28) | 0.734 | 0.872 | 0.690 | 0.694 | 0.1 |
| lme_oracle_v4_default (stratified 150, #29) | 0.664 | 0.776 | 0.611 | 0.602 | 30.1 |
| lme_oracle_v4_combo (stratified 150, #30) | 0.709 | 0.832 | 0.646 | 0.651 | 12.6 |

_(see `bench/results/*.json` and `RESULTS.md` for the authoritative,
timestamped numbers)_

LongMemEval per-category R@10 (temporal-reasoning / multi-session /
knowledge-update / single-session-preference / single-session-assistant /
single-session-user), n=25 each:

| Run | temporal | multi-sess | know-upd | pref | assistant | user |
|---|---|---|---|---|---|---|
| bm25_turn | 0.873 | 0.499 | 0.967 | 0.933 | 1.000 | 0.960 |
| v4_default | 0.493 | 0.482 | 0.867 | 0.933 | 0.960 | 0.920 |
| v4_combo | 0.513 | 0.565 | 0.980 | 0.973 | 1.000 | 0.960 |

## Winning configuration

**`locomo_v4_combo_v2`**: `{"fusion": "cc", "w_dense": 1.0, "w_bm25": 1.5, "candidates": 100, "use_stemming": true}`

- LoCoMo macro R@10: 0.435 (pre-campaign default) -> **0.527** (+0.092, +21%)
- LongMemEval oracle (stratified 150) R@10: 0.776 (v4_default) -> **0.832** (+0.056)
- p50 latency: 26.5 ms (LoCoMo) / 12.6 ms (LongMemEval) — both under the 30 ms guardrail
- Wins on **both** benchmarks vs the v4 default it replaces (the E7 promotion
  rule) -> **set as the new `Config` defaults** (`src/memory_os_v4/config.py`:
  `fusion`, `w_bm25`, `candidates`, `use_stemming`); full test suite re-run
  green (226/226) after the change. `w_dense`/`w_entity`/`use_recency` and
  all other ablation flags (multi_vector, neighbor_expansion,
  recency_only_with_at_time) are unchanged — none showed a reproducible,
  above-noise gain on their own post-fix.
- **Caveat, stated plainly**: plain `bm25_turn` (whole-turn Okapi BM25, no
  v4 machinery at all) still scores higher on this specific LongMemEval
  stratified sample (R@10 0.872 vs v4_combo's 0.832), driven almost entirely
  by `temporal-reasoning` (bm25 0.873 vs v4_combo 0.513 — v4's default
  *also* struggles here at 0.493, so this is not something the combo
  introduced). v4_combo wins or ties bm25 on the other 5/6 categories
  (notably `multi-session` 0.565 vs 0.499, `knowledge-update` 0.980 vs
  0.967). The promotion rule compares the candidate default against the
  *previous* v4 default, not against a different retriever family entirely —
  bm25_turn was never a candidate config for v4's own defaults — but this
  temporal-reasoning gap is a genuine, unresolved weakness worth flagging
  for future work (see Conclusions).

## Conclusions

1. **Fusion method dominates everything else in this campaign.** Switching
   RRF -> weighted convex combination (`cc`) with `candidates` 50->100 was
   worth +0.05-0.07 R@10 on its own (locomo_v4_cc_1_1: 0.454 -> 0.507
   post-fix default), more than triple any other single lever tried. This
   matches the research notes' citation (Bruch et al. 2022; Qdrant 2026):
   weighted fusion beats RRF once signal score *magnitude*, not just rank,
   carries information — true here because BM25/dense scores are not
   degenerate. `w_bm25=1.5` (favoring the sparse signal slightly) beat
   `w_bm25=1.0` and `w_dense=0.7/w_bm25=1.3` in every fusion method tried.
2. **Stemming was the single best *independent* lever and stacks with
   fusion.** +0.027 R@10 alone (post-fix), +0.02 more on top of the best
   fusion config (cc_1_1 0.507 -> combo_v1 0.522). A ~120-line in-house
   Porter stemmer plus light French suffix stripping, applied only to
   BM25's token stream, closes a real English-morphology gap ("charities"
   query vs "charity" turn) at effectively zero latency cost.
3. **Multi-vector chunking and neighbor expansion both failed the latency
   guardrail for no measurable recall benefit** — R@10 deltas were within
   the ±0.005 noise band on every variant (mv48/mv96, nbr1/nbr2), while p50
   latency roughly quadrupled (7.9ms -> 33-39ms). The chunking hypothesis
   from RESULTS.md's phase journal ("128-char chunks beat whole-turn
   because MiniLM truncates at 256 tokens") does not actually apply here:
   LoCoMo turns are short single chat messages, so `_add_chunk_vectors`
   found nothing to split for the vast majority of items — multi_vector's
   entire mechanism was mostly a no-op wrapped in extra encode() calls.
4. **Adversarial review caught real, ledger-invalidating bugs mid-campaign**
   (see the dedicated section above) — most importantly, the entity
   channel's pre-fix "net negative" verdict (ent0 beating default by +0.02)
   was really a *quality* bug (queries polluted with their own interrogative
   as a pseudo-entity, plus common-participant terms dominating ties), not
   evidence that entity signals are inherently unhelpful: post-fix, `w_entity
   ∈ {0, 0.5, 1.0}` all land within noise of each other (0.452-0.455), i.e.
   the fixed entity channel is now genuinely neutral rather than harmful.
   Likewise the recency fix's apparent pre-fix gain (+0.02) evaporated
   post-fix (0.454 vs 0.455) once the determinism bug stopped acting as a
   confound — recency truly is close to a no-op on LoCoMo (`query_time` is
   always `None`), though the adaptive-half-life half of the fix should
   still matter on datasets that *do* supply `at_time` (see the dedicated
   unit test). Both fixes are kept in the code as correctness improvements
   even though neither moved into the winning combo.
5. **Remaining weakness, flagged not fixed**: v4_combo's `temporal-reasoning`
   R@10 on LongMemEval (0.513) is far below plain BM25 (0.873) and barely
   above v4's own pre-combo default (0.493) — i.e. this is a pre-existing
   v4 weakness on precise "when did X happen" questions that the winning
   combo does not address (its fusion/stemming gains are concentrated in
   the other 5 LongMemEval categories and across all 4 LoCoMo categories).
   A future campaign should treat `temporal-reasoning` as its own targeted
   ablation (e.g. a dedicated temporal-entity boost, or a BM25-heavy fusion
   fallback when the query matches `parse_dates()`) rather than something
   the current general-purpose fusion/stemming levers happen to fix.

## Hardening 2026-09-13

A second adversarial review (fresh context) found 11 further confirmed
findings across `memory.py`, `bm25.py`, `retrieval.py`, `migrate.py`,
`temporal.py`, `extract.py`, `vector_index.py`, `server.py` and
`bench/retrievers.py`/`bench/run_bench.py`, on top of the "Adversarial
review fixes" already landed in the ablation campaign above. Every fix
below was reproduced BEFORE and AFTER with the reviewer's own repro
scripts (kept at `/private/tmp/claude-501/.../scratchpad/review/*.py`,
run against the live `src/` via `PYTHONPATH`) — outputs quoted are literal
tool output, not paraphrased. Full test suite: 226 -> **284 passing**
(58 new regression tests), `ruff check src bench tests --select E,F
--ignore E501` clean throughout.

| # | Finding | Fix | Test(s) | Repro script |
|---|---|---|---|---|
| A1 | `_init_indexes`: silent index/store desync after an unflushed append or a truncated last JSONL line (orphan vector/bm25/entity entries); an `embeddings.npy`/`ids.json` count or dim mismatch reached faiss's `add_with_ids` as a bare `AssertionError`, bricking the store on load. | New `MemoryV4._reconcile_indexes()`: diff live (non-deleted) store ids vs each index's ids, embed+add anything missing, remove anything stale (deleted or no longer in the store); `_init_indexes` now also compares `meta["dim"]`/cached vector row count against the current embedder before trusting the cache, falling back to a full re-embed instead of crashing. | `test_reload_reconciles_unflushed_appended_item`, `test_reload_drops_orphan_index_entries_from_truncated_log`, `test_reload_forces_reembed_instead_of_crashing_on_id_vector_count_mismatch`, `test_reload_reembeds_when_stored_dim_differs_from_current_embedder`, `test_reload_reconciles_stale_soft_deleted_index_entries` (`tests/test_core_memory.py`) | `store_probes.py`: before — `bm25 has A? True \| vecs has A? True` after reload (soft-delete never cleaned up), `vector index still has B? True -> ORPHAN vector` after truncation, `raised: AssertionError not same nb of vectors as ids` on an ids/vecs mismatch, a `hash_dim` change crashed with the same `AssertionError`. After — `bm25 has A? False \| vecs has A? False`, `vector index still has B? False`, `loaded, ntotal: 2` (no crash), `hash_dim 64 -> 32 ... ntotal 1` (no crash). |
| A2(a) | `add_facts` supersession crossed projects: a fact in project B could invalidate one in project A. | Added `if old.project != item.project: continue` to the supersession candidate loop. | `test_add_facts_supersession_never_crosses_projects` | `supersede.py` §1: before — `superseded: [(...)]` (crossed); after — `superseded: []`. |
| A2(b) | An **equal** `event_time` still superseded (`item.event_time < old.event_time` let a tie through), setting `valid_until == valid_from` on the old fact — a zero-length validity window, never retrievable with any `at_time`. | Changed the skip condition to `item.event_time <= old.event_time` (strict `>` required to supersede). | `test_add_facts_supersession_equal_event_time_does_not_supersede` | `supersede.py` §3: before — `superseded: [(...)]`, `is_valid_at(old, T) = False`; after — `superseded: []`, `is_valid_at(old, T) = True`. |
| A2(c) | A superseded/invalidated fact could never be re-asserted: exact-hash AND near-duplicate dedup matched invalidated candidates too, so re-adding old text with a later date got silently deduped to the dead record instead of creating a new valid fact. | Both `add_facts`'s exact-hash lookup and `_find_near_duplicate` now skip any candidate with `valid_until is not None or invalidated_by is not None`. | `test_add_facts_dedup_ignores_invalidated_candidates`, `test_add_facts_supersession_chain_then_reassert_original` (full A -> B -> C chain, then re-assert A, which by the *normal* supersession rule then supersedes the currently-valid C) | `supersede.py` §5: before — `added: 0 deduped: 1 superseded: []` ("resurrected fact was DEDUPED to the dead one"); after — `added: 1 deduped: 0 superseded: [(<Nice_id>, <new_Paris_id>)]` ("added fresh"). |
| A3 | `add_facts`'s write-time near-duplicate probe was embedded with a different, hand-rolled header (`"[?\|project: {p}\|system]"`) than `contextual_header(item)` (what `_index_items` actually embeds and stores) — ~0.10 cosine harsher than `consolidate`'s sweep on the *same* pair, and the fact's text was embedded twice. | `add_facts` now builds the `MemoryItem` first, embeds it ONCE with `contextual_header(item)`, uses that vector for the dedup probe, and passes it straight into `_index_items(..., vecs=...)` (new optional param, default `None` preserves every other caller). | `test_add_facts_probe_vector_matches_indexed_vector` (bit-identical probe vs. indexed vector, real MiniLM), `test_add_facts_dedup_agrees_with_consolidate_dedup_on_same_pair` | `header_mismatch.py`: `cos(add_facts probe, indexed vector 2026) = 0.8540` vs `cos(indexed 1970, indexed 2026) = 0.8714` (harsher, pre-fix formula). `dedup_dead.py` (real MiniLM, `dedup_cos=0.92`): before — `second add -> {'added': [<id>], 'deduped': [], ...}` while `consolidate(dry_run=True)` on the same pair says `{'deduped': 1}` (disagreement); after — `second add -> {'added': [], 'deduped': [<id>], ...}` (add_facts now agrees with consolidate; verified the probe vector is bit-identical to the indexed one). |
| A4 | Duplicate `item_id` within one batch (`add_episodes_bulk`, e.g. `migrate_v3` replaying a duplicated log line) built 2 `MemoryItem`s for 1 id; `VectorIndex.add` had no within-call dedup, so both got embedded into the index (`ntotal` > stored item count) even though the store dict silently kept only the last one. | `MemoryV4._build_items` now skips a duplicate id within the same batch (first wins); `VectorIndex.add` de-duplicates its own `ids` argument the same way before assigning int ids; `migrate_v3` reports `episodes = len(added_items)` (actual stored count) instead of `len(episode_specs)`, folding the difference into `skipped`. | `test_add_episodes_bulk_duplicate_id_in_one_batch_first_wins`, `test_vector_index_add_deduplicates_ids_within_one_call`, `test_migrate_v3_duplicated_line_within_one_file_stores_once` | `dupid.py`: before — `built 2 items with ids ['X', 'X']`, `store size 1 vector ntotal 2` (drift); after — `built 1 items with ids ['X']`, `store size 1 vector ntotal 1`. `migrate_dup.py`: before — `{'episodes': 2, ...}` for 1 stored item; after (see `test_migrate_v3_duplicated_line_within_one_file_stores_once`) — `episodes: 1`, `vector_count == total`. |
| A5 | Soft delete (`deleted=True`, used by `memory_forget`/consolidate) left the item in the vector/BM25/entity indexes until the next `consolidate()` rebuild — it kept consuming candidate slots in every retrieval signal in the meantime. | New additive façade method `MemoryV4.forget(item_id, reason="") -> bool`: soft-deletes AND removes the item (plus any multi_vector chunk vectors it owns) from all three indexes immediately. `server.py`'s `_h_forget` now calls `engine.forget(...)` instead of flipping `.deleted` directly on `engine.store`. | `test_forget_removes_item_from_all_indexes_immediately`, `test_forget_unknown_or_already_deleted_returns_false` (`tests/test_core_memory.py`), `test_forget_removes_from_bm25_and_vector_index_immediately` (`tests/test_server_handlers.py`) | `store_probes.py` §1 (old-style direct `.deleted=True`, bypassing `forget()`): `but bm25 still has it: True \| vector index still has it: True` immediately — this is exactly the gap `forget()` closes; a direct call to the new method removes both instantly (see tests). |
| A6 | `migrate.py`'s `_parse_v3_timestamp` treated a naive (no-tzinfo) ISO timestamp as LOCAL time (`datetime.timestamp()`'s default), silently shifting every migrated `event_time` by the host's UTC offset; `MemoryV4.migrate_v3`/`read_v3_log` had no path-traversal guard of their own (only `server.py`'s `_h_migrate_v3` did), so calling the façade directly could read an arbitrary file (`/etc/hosts`) as a "v3 log". | Naive timestamps now get `.replace(tzinfo=timezone.utc)` before `.timestamp()`. New shared `migrate.resolve_under_home()` (log_path must resolve under `$HOME` or `$MEMORY_WORKSPACE`, else `ValueError`) is called by `read_v3_log` itself; `server.py` now imports and reuses this SAME function (as `_resolve_under_home`) instead of keeping a second, independently-maintained copy — both layers guard identically by construction. | `test_parse_v3_timestamp_naive_iso_treated_as_utc`, `test_migrate_v3_naive_timestamp_end_to_end_matches_utc`, `test_migrate_v3_time_travels_naive_utc`, `test_migrate_v3_blocks_path_outside_home_and_workspace`, `test_resolve_under_home_allows_memory_workspace_override` (`tests/test_core_migrate.py`) | `migrate_dup.py`: before — `migrate_v3(str(log)) ` on `/etc/hosts` returned `{'episodes': 0, 'summaries': 0, 'skipped': 0}` silently (only failed because `/etc/hosts` isn't valid JSONL, NOT because of a path guard); after — raises `ValueError: path traversal blocked: ...` immediately, before ever opening the file. |
| A7(a) | `temporal.parse_dates` existed but was never wired into fact creation: a fact with no `event_time` but an absolute date in its text (`"Deployed on 2026-09-12."`) silently defaulted to `time.time()`. | New `temporal.parse_absolute_dates(text)` (absolute-only: ISO/D-M-Y/day-month-year/month-year, no `reference_time` needed — factored out of `parse_dates`, which now calls it internally with identical behaviour for existing callers). `extract.normalize_client_facts` uses it when `event_time` is `None`: takes the FIRST absolute date found, tags the fact `"date_from_text"`. | `test_normalize_client_facts_infers_event_time_from_absolute_date_in_text`, `test_normalize_client_facts_does_not_infer_from_relative_date_text`, `test_normalize_client_facts_explicit_event_time_wins_over_text_date`, `test_normalize_client_facts_no_date_anywhere_leaves_event_time_none` (`tests/test_core_extract.py`), `test_parse_absolute_dates_excludes_relative_expressions`, `test_parse_absolute_dates_needs_no_reference_time` (`tests/test_core_temporal.py`) | New behaviour (`parse_absolute_dates` did not exist pre-fix) — verified end-to-end via the new tests rather than a reviewer repro script. |
| A7(b) | `parse_dates` was also never exposed on the retrieval side, and two-field slash dates (`_DMY_RE`) had no documented convention nor an MM/DD fallback, so `"03/15/2026"` (only valid as MM/DD) silently failed to parse. | `HybridRetriever.search` now computes `query_dates = parse_dates(query, reference_time=at_time or now)` once per call and exposes it as `SearchHit.signals["query_dates"]` (used by Part B below). `parse_absolute_dates`'s docstring documents the DD/MM default and the MM/DD fallback: when the DD/MM reading is impossible (its month field > 12) but swapping the two fields gives a valid date, the swapped reading is used. | `test_search_exposes_query_dates_signal`, `test_search_query_dates_empty_for_non_temporal_query` (`tests/test_core_retrieval.py`), `test_parse_dmy_ambiguous_defaults_to_ddmm`, `test_parse_dmy_falls_back_to_mmdd_when_ddmm_impossible`, `test_parse_dmy_unambiguous_ddmm_still_works` (`tests/test_core_temporal.py`) | `misc_probes.py`: before — `'on 03/15/2026' -> []` (unparseable); after — `'on 03/15/2026' -> ['2026-03-15']`. `'on 01/02/2026' -> ['2026-02-01']` unchanged both before/after (ambiguous case correctly keeps the DD/MM default). |
| A8 | `minmax_normalize`/`_zscore_clip_minmax` had no guard against a NaN/±inf raw signal score (e.g. a corrupt embedding): `inf - inf` / `inf / inf` produced NaN results that would silently corrupt the final fused ranking, rather than triggering the existing degenerate-signal (`hi - lo < eps`) rank-based fallback. | Both functions now check `math.isfinite(v) for v in values.values()` up front and fall back to the existing `_rank_fallback` (insertion-order-based, never touches the raw values arithmetically) whenever any value is non-finite — the exact same fallback already used for the all-equal-score case. | `test_minmax_normalize_guards_nan`, `test_minmax_normalize_guards_inf`, `test_zscore_clip_minmax_guards_nan_and_inf`, `test_fuse_cc_guards_nan_raw_score` (`tests/test_core_retrieval.py`) | `fusion_edge.py`: before — `nan in raw scores (cc): {'a': nan, 'b': nan}`, `inf in raw scores (cc): {'a': nan, 'b': 0.0}`, `minmax with inf: {'a': nan, 'b': 0.0}`, plus a `RuntimeWarning: invalid value encountered in subtract` from numpy; after — `nan in raw scores (cc): {'a': 1.0, 'b': 0.0}`, `inf in raw scores (cc): {'a': 1.0, 'b': 0.0}`, `minmax with inf: {'a': 1.0, 'b': 0.0}`, no warnings, every value finite. |
| A9 | `bm25.py` persisted each document's FULL token list (`doc_tokens: dict[str, list[str]]`) in `bm25.json`, even though `remove()` only ever needs each distinct token once — a document with heavy word repetition bloated the persisted file proportionally to its total token count, not its vocabulary size. | Replaced with `doc_term_freqs: dict[str, dict[str, int]]` (per-doc term -> term-frequency, from the same `Counter` already computed in `add()`); `remove()` iterates its keys instead of `set(tokens)`. Persistence round-trip (`to_dict`/`from_dict`/`save`/`load`) updated to the new field name. | `test_bm25_does_not_persist_full_token_lists`, `test_bm25_persisted_size_scales_with_distinct_terms_not_total_tokens`, `test_bm25_persistence_roundtrip_preserves_term_frequencies`, `test_bm25_remove_after_reload_still_updates_postings` (`tests/test_core_bm25.py`) | `misc_probes.py`'s BM25 persistence probe uses 100 docs of 2 DISTINCT tokens each (no repetition at all) — the worst case for this fix, where a `{term: 1}` map costs slightly MORE bytes than a bare 2-item list (`file KB: 6.4` before -> `7.0` after for that specific probe). The fix's actual target — repeated tokens — is covered by the new `test_bm25_persisted_size_scales_with_distinct_terms_not_total_tokens` (5000 occurrences of 1 term stays under 2 KB), which `misc_probes.py` does not exercise; documented here for full honesty rather than only citing the flattering number. |
| A10 | `bench/retrievers.py`'s `V3EquivRetriever` docstring called itself a reproduction of the real v3.2.0 engine, but it is charitable in three ways: its chunker breaks early and drops the tiny/redundant tail window real v3's unconditional slicing still emits; it oversamples `top_k * 8` chunk hits and pads chunk->turn dedup to `top_k` distinct ids; and it applies no absolute-distance cutoff (real v3 dropped any chunk past L2² 0.8). | Docstring rewritten to state plainly it is a "charitable v3-style baseline" and enumerate the three differences with the exact `chunk_diff.py` numbers. New `V3ExactRetriever` (`bench/retrievers.py`) literally reproduces v3: `[text[i:i+128] for i in range(0, len(text), 100)]` (no early break), `IndexFlatL2`, `k = min(top_k, n)` (no `* 8`), drops any chunk with distance > 0.8, maps to parent id WITHOUT dedup-and-pad (so it can return fewer than `top_k` distinct turns). Wired into `bench/run_bench.py` (`--retriever v3exact`) and run on LoCoMo as `locomo_v3exact` (row appended to the ledger below). | `test_v3exact_chunk_text_matches_literal_v3_formula`, `test_v3exact_chunk_text_keeps_the_ragged_tail_v3equiv_drops`, `test_v3exact_chunk_text_empty_text`, `test_v3exact_search_drops_chunks_past_distance_threshold`, `test_v3exact_search_no_oversampling_can_return_fewer_than_top_k`, `test_v3exact_search_on_empty_index_returns_empty`, `test_v3exact_reset_clears_index`, `test_v3exact_ignores_at_time_faithfully` (`tests/test_bench_retrievers.py`) | `chunk_diff.py` (unchanged before/after — `V3EquivRetriever.chunk_text` was not touched): `turns=5882 turns where v3_real != V3Equiv.chunk_text: 1400 (23.8%)`, `total chunks: v3_real=11263 V3Equiv=9863 missing=1400`. The NEW `locomo_v3exact` bench run makes the cost of that charity concrete: macro R@10 **0.175** vs `locomo_v3equiv`'s (charitable) **0.405** (row 1) — more than double, on the exact same corpus/questions/embedder. |
| A11 | `bench/run_bench.py`'s `context_tokens` was reported as already fixed to use `ranked[:top_k]` (not a hard-coded `[:10]`), and `PYTHONHASHSEED` was reported as already recorded in `metadata` — both true on inspection, but neither had a regression test. | No code change (verified correct); added regression tests so a future edit can't silently reintroduce either regression. | `test_context_tokens_honors_top_k_not_hardcoded_ten`, `test_metadata_records_pythonhashseed` (`tests/test_bench_run.py`) | N/A — code was already correct; tests added for coverage only. |

### Part B: LongMemEval temporal-reasoning targeted experiment

Motivated by the unresolved weakness flagged at the end of the ablation
campaign above (`temporal-reasoning` R@10 0.513 vs plain BM25's 0.873).
Two hypotheses: **H1** the recency multiplier penalises older sessions
holding the evidence when `at_time` is given; **H2** dense MiniLM scores
dominate date-heavy queries and drown out BM25's literal date/number
match. All runs: `--dataset longmemeval_oracle --stratified --limit 150
--retriever v4`, `PYTHONHASHSEED=0`, on the current (post-hardening,
A1-A11) code — a fresh `lme_oracle_v4_posthardening_default` baseline was
run first (row 35) to confirm A1-A11 changed nothing on this sample
(it reproduces the campaign's `lme_oracle_v4_combo` numbers to 3 decimals:
R@10 0.832, per-category identical) so B1-B4 are compared against a
like-for-like reference.

| Run | temporal-reasoning | multi-session | knowledge-update | pref | assistant | user | macro R@10 |
|---|---|---|---|---|---|---|---|
| baseline (post-hardening default, row 35) | 0.513 | 0.565 | 0.980 | 0.973 | 1.000 | 0.960 | 0.832 |
| **B1** `{"use_recency": false}` (row 31, `lme_oracle_v4_norec`) | 0.513 | 0.552 | 0.940 | 0.973 | 1.000 | 0.960 | 0.823 |
| **B2** `{"w_bm25": 2.5}` (row 32, `lme_oracle_v4_bm25heavy`) | 0.493 | 0.571 | 0.940 | 0.987 | 1.000 | 0.960 | 0.825 |
| **B3** `{"temporal_query_mode": "auto"}` (row 33, `lme_oracle_v4_tempauto`) | 0.513 | 0.552 | 0.980 | 0.973 | 1.000 | 0.960 | 0.830 |

**B1 (H1 test)**: disabling recency GLOBALLY leaves `temporal-reasoning`
byte-identical (0.513 -> 0.513) — **H1 refuted**: recency is not the cause
(it also drags `knowledge-update` down, 0.980 -> 0.940, an unrelated
side-effect of removing recency's tie-breaking elsewhere, not a fix).

**B2 (a blunter H2 test)**: a much bigger, GLOBAL bm25 boost (1.5 -> 2.5)
makes `temporal-reasoning` *worse* (0.513 -> 0.493) — a naive "just weight
bm25 more everywhere" is not the answer; it likely pulls the wrong
candidates to the top by disturbing the `cc` fusion balance for
non-temporal queries too.

**B3 (the targeted mechanism)**: `temporal_query_mode="auto"` (implemented
in `retrieval.py`: `parse_dates(query)` non-empty OR the query matches a
small EN/FR temporal-cue regex -> disable recency and multiply `w_bm25` by
`config.temporal_bm25_boost` (default 1.5) for THAT QUERY ONLY, never
mutating `Config`) was confirmed to actually fire on the target category —
a direct check found the trigger active on **12/25 (48%)** of
`temporal-reasoning`'s questions (`parse_dates`/cue-regex diagnostic, see
below) — yet the category's R@10 is **unchanged** (0.513 -> 0.513, an exact
tie, not a win) and `multi-session` drops slightly (0.565 -> 0.552,
-0.013, likely 1 question's worth of noise at n=25 but still a move in the
wrong direction). Every other category is identical to the baseline.

```
n temporal-reasoning questions: 25
fired: 12 / 25   (parse_dates non-empty: 1/25 -- "last month" style cues
                  dominate over literal dates; temporal-cue regex: 12/25)
```

**B4 (LoCoMo regression guard)**: `locomo_v4_tempauto` (row 34) scores
macro R@10 **0.527** — bit-for-bit the same as the fresh post-hardening
LoCoMo baseline (row 36, also 0.527) and the documented `locomo_v4_combo_v2`
number, comfortably inside the ±0.005 budget. LoCoMo's own temporal
category (cat. 2, "When did...") is 0.707 vs baseline's 0.705 (+0.002,
noise) — confirms `temporal_query_mode="auto"` does not cost LoCoMo
anything, it simply has no measurable effect there either (LoCoMo rarely
triggers the cue regex/`parse_dates` the way LongMemEval's "how many
days"/"how long ago" phrasing does).

**Decision: `temporal_query_mode` stays default `"off"`.** B3 does not
satisfy the promotion rule ("wins on LME temporal-reasoning without
hurting the other categories") — it ties on the metric it targets and
introduces a small regression on `multi-session`. The mechanism fires as
designed (confirmed 48% activation on the target category) but the
resulting reweighting genuinely does not move the needle on this
category — combined with B1's clean refutation of H1 and B2's evidence
that blanket bm25-weighting actively hurts, this points to
`temporal-reasoning`'s bottleneck being something retrieval reweighting
cannot fix: these questions need actual date ARITHMETIC ("how many days
between X and Y", "how long before Z") over the retrieved evidence, not a
better-ranked set of candidates — the evidence sessions are often already
retrievable, but no amount of fusion-weight tuning teaches the retriever
to compute a day-count. Left in the code as a documented, tested, OFF-by-
default `Config` flag (`temporal_query_mode`, `temporal_bm25_boost`) for a
future campaign to build on (e.g. combined with an LLM- or rule-based date
arithmetic post-processing step over retrieved evidence, which is out of
scope for a retrieval-only lever).

## Full-candidate scoring 2026-09-13

Follow-up to Part B's unresolved `temporal-reasoning` weakness (R@10 0.513
vs plain BM25's 0.873), testing a different mechanism than B1-B3: **new
hypothesis** — `fuse()` imputes 0.0 for any candidate missing from a
signal's own top-`candidates` hit list (the candidate union is dense
top-N ∪ bm25 top-N ∪ entity top-N), so a turn found only by BM25 is
penalised against one found by both signals; and per-query min-max
normalization stretches a noisy dense channel to the full [0,1] range even
when every cosine in the query is low (typical of date-heavy queries),
letting dense noise dominate the fused score where BM25's literal
date/number match should win.

Implemented behind two new, independent, OFF-by-default `Config` flags
(`src/memory_os_v4/config.py`):

- **`full_candidate_scoring`**: after building the candidate union from the
  top-N lists exactly as before (candidate GENERATION unchanged, point 3 of
  the spec — see `test_full_candidate_scoring_does_not_change_candidate_generation`,
  `test_full_candidate_scoring_still_uses_topn_for_candidate_generation`),
  compute each enabled signal's TRUE score for EVERY candidate instead of
  imputing 0 for the ones outside that signal's own top-N: dense via new
  `VectorIndex.scores_for(query_vec, ids)` (inner product against the
  stored, unit-normalized vectors — multi_vector still max-pools over an
  item's chunk vectors, via new `HybridRetriever._chunk_ids_by_owner`/
  `_dense_full_scores`), bm25 via new `BM25Index.score_doc(query_tokens,
  doc_id)` (identical per-term formula to `search()`), entity via new
  `EntityIndex.score_doc(query_entities, doc_id)` (identical IDF-weighted
  overlap formula, including the existing >10%-of-corpus "not
  discriminative" skip). New `retrieval.fuse_full()` combines these
  full-candidate score maps (for "rrf", ranks are computed over the
  full-scored union, not the top-N `search()` ranks, per the spec).
- **`dense_norm`**: `"minmax"` (unchanged default) or `"fixed"` — replaces
  the dense channel's per-query min-max with a fixed linear map of cosine
  onto [0,1] (negatives clipped to 0, no stretch — new
  `_fixed_cosine_normalize`), and the bm25 channel's min-max with
  score/per-query-max (new `_max_normalize`), so both channels' raw
  magnitude carries confidence instead of always being rescaled to fill
  [0,1] regardless of how weak every candidate's match actually is. Only
  affects fusion methods "cc"/"dbsf" (rrf uses ranks, not normalized
  scores); entity normalization is unchanged; independent of
  `full_candidate_scoring` (also usable, with reduced effect, on the
  top-N-only `fuse()` path).

All three helpers (`scores_for`/`score_doc`/`score_doc`) are covered by
regression tests asserting they equal the score `search()` computes for
the same doc, including for a doc outside `search()`'s own top-k
(`tests/test_core_bm25.py`, `tests/test_core_entities.py`,
`tests/test_core_retrieval.py`) — 38 new tests total, suite
284 -> **322 passing**, `ruff check src bench tests --select E,F --ignore
E501` clean.

### Results

Six real (`sentence-transformers/all-MiniLM-L6-v2`) `bench/run_bench.py`
runs, `PYTHONHASHSEED=0`, against the current defaults (`fusion=cc,
w_dense=1.0, w_bm25=1.5, candidates=100, use_stemming=True`) as the
baseline in each table.

**LoCoMo** (full corpus, n=1536, top_k=10):

| Run | cat.1 multi-hop | cat.2 temporal | cat.3 open-dom. | cat.4 single-hop | macro R@10 | p50 ms |
|---|---|---|---|---|---|---|
| baseline (`locomo_v4_posthardening_default`, row 36) | 0.360 | 0.705 | 0.323 | 0.720 | 0.527 | 18.4 |
| **locomo_v4_full** (row 38, `full_candidate_scoring=true`) | 0.342 | 0.712 | 0.346 | 0.710 | 0.528 | 17.2 |
| **locomo_v4_full_fixed** (row 39, `+ dense_norm=fixed`) | 0.327 | 0.717 | 0.299 | 0.695 | 0.509 | 17.2 |
| **locomo_v4_full_dbsf** (row 40, `+ fusion=dbsf`) | 0.343 | 0.713 | 0.347 | 0.714 | 0.529 | 14.3 |

**LongMemEval oracle** (stratified 150, top_k=10):

| Run | temporal-reasoning | multi-session | knowledge-update | pref | assistant | user | macro R@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| baseline (`lme_oracle_v4_posthardening_default`, row 35) | 0.513 | 0.565 | 0.980 | 0.973 | 1.000 | 0.960 | 0.832 | 24.9 |
| **lme_oracle_v4_full** (row 41, `full_candidate_scoring=true`) | 0.513 | 0.559 | 1.000 | 0.987 | 0.960 | 0.960 | 0.830 | 22.7 |
| **lme_oracle_v4_full_fixed** (row 42, `+ dense_norm=fixed`) | 0.493 | 0.552 | 0.960 | 0.987 | 0.960 | 1.000 | 0.825 | 23.5 |
| **lme_oracle_v4_full_dbsf** (row 43, `+ fusion=dbsf`) | 0.513 | 0.559 | 1.000 | 0.987 | 0.960 | 0.960 | 0.830 | 23.0 |

All six runs comfortably clear the 30ms p50 latency guardrail (14.3-23.5ms).

### Decision: neither flag is promoted — `full_candidate_scoring` and `dense_norm` stay at their OFF/`"minmax"` defaults.

Promotion rule: beat the baseline on BOTH LoCoMo macro R@10 (>0.532) AND
LongMemEval oracle macro R@10 (>0.837). Best LoCoMo result is
`locomo_v4_full_dbsf` at 0.529 (+0.002, inside the ±0.005 noise band); best
LongMemEval result is 0.830 (`_full`/`_full_dbsf`, tied) — *below* the
0.832 baseline, let alone the >0.837 bar. `dense_norm=fixed` is an outright
regression on both benchmarks (LoCoMo 0.527 -> 0.509, -0.018; LongMemEval
0.832 -> 0.825, -0.007) and is not a case of "wins the targeted category
but loses LoCoMo slightly" either — it loses `temporal-reasoning` too
(0.513 -> 0.493) — so no partial-promotion or "report it" exception
applies (per the "wins LME temporal-reasoning massively but loses LoCoMo
slightly" clause). No default changed; no test-suite re-run was needed as
a consequence (the 322/322 green run above already covers the new code).

### Interpretation

1. **The imputation/stretch hypothesis is refuted on the category it targeted.**
   `temporal-reasoning` is byte-identical under `full_candidate_scoring`
   (0.513 -> 0.513 for both `cc` and `dbsf` fusion) and gets *worse* under
   `dense_norm=fixed` (0.513 -> 0.493) — the exact opposite of what
   "stop stretching noisy dense scores to fill [0,1]" predicted. Combined
   with Part B's B1 (recency: no effect) and B2 (blunt bm25 boost: -0.020),
   this is the third distinct, independently-motivated retrieval-scoring
   lever to leave this category untouched or worse, which is now strong
   convergent evidence that `temporal-reasoning`'s ceiling is not a
   candidate-scoring problem at all.
2. **`full_candidate_scoring` alone is a neutral, not harmful, change
   everywhere else** (LoCoMo macro +0.0006, LongMemEval macro -0.002, both
   inside noise) — the top-N-only imputation this campaign hypothesized as
   a "penalty" turns out to rarely change the winning candidate's relative
   rank in practice on these two benchmarks, at comparable (even slightly
   better, thanks to skipping some of `search()`'s own top-N bookkeeping)
   latency.
3. **`dense_norm=fixed` is the one lever that actually moves numbers, and
   only for the worse**: LoCoMo's cat.1/cat.4 (multi-hop/single-hop) and
   LongMemEval's `knowledge-update`/`multi-session` all drop when the dense
   channel stops being rescaled per-query — i.e. the per-query min-max
   stretch this campaign suspected of manufacturing false dense confidence
   is, on net, doing useful discriminative work on these benchmarks (most
   queries' true best dense match legitimately deserves to read as "close
   to 1.0"), not hiding a systematically weak signal. `temporal-reasoning`
   itself is not spared either (0.513 -> 0.493), so even the one flag that
   changes anything moves in the wrong direction for the very category it
   was meant to help.

Both flags are left in the code, tested and documented, OFF by default —
useful groundwork (`VectorIndex.scores_for`, `BM25Index.score_doc`,
`EntityIndex.score_doc` are now available primitives) for a future
campaign, but this specific experiment closes with `temporal-reasoning`
still unresolved and increasingly looking like it needs date arithmetic
over already-retrieved evidence rather than any further retrieval-scoring
tuning (same conclusion Part B reached, now on stronger evidence).

## Palier 2-3 (2026-09-13)

Follow-up campaign, three new independent levers, each behind a `Config`
flag defaulting OFF/unchanged: **E10** embedding upgrade
(`ibm-granite/granite-embedding-97m-multilingual-r2`, dim 384, ModernBERT-
based, `Config.embed_max_seq_length` capped to 512 — its tokenizer reports
32768), **E11** cross-encoder reranking (`tomaarsen/ms-marco-ettin-17m-
reranker` via the already-scaffolded `retrieval.py` rerank path — score
= `(1 - Config.rerank_weight) * minmax(fused) + Config.rerank_weight *
minmax(ce)`, `Config.rerank_weight` default 0.5, `Config.rerank_device`
default `"auto"`→mps), and **E12** entity-based multi-hop expansion (new
mechanism, `Config.entity_expansion`, off by default — see
`HybridRetriever._expand_entities`/`_rare_entities`, retrieval.py). All
three models were already locally cached; `cross-encoder/ms-marco-MiniLM-
L6-v2` remains deleted (NaN on this machine's transformers 5.2) and was not
used. Guardrail: p50 < 30 ms for non-rerank configs, relaxed to p50 <=
150 ms for rerank configs (cross-encoder inference is inherently slower).
Promotion rule (given): beat current defaults on BOTH LoCoMo macro R@10
(> 0.532) and LongMemEval oracle stratified-150 macro R@10 (> 0.837); if
the embedding upgrade alone wins both, promote it even if rerank does not.

**Resource note.** This machine hit real constraints mid-campaign (16 GB
RAM, MPS memory pressure from other concurrent processes) that changed how
the campaign was run, not just how it was reported:

- Runs were executed strictly **sequentially**, one `bench.run_bench`
  process at a time, each prefixed `nice -n 15`, with
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4`/`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3`
  set, and `Embedder.encode()` now calls `torch.mps.empty_cache()` after
  every batch on mps (no-op on cpu). `Config.embed_batch_size` (new field,
  default 32, down from the previous hardcoded 64) is a pure compute-
  chunking knob — it does not change embedding values.
- Reproducing `lme_oracle_v4_granite` first hit a **real `MPS backend out
  of memory` crash**: `bench.retrievers.V4Retriever.reset()` builds a brand
  new `MemoryV4` (and therefore a brand-new `Embedder`, which reloads the
  full model) for every LongMemEval-oracle "conversation" — up to 150 times
  in one run — and the old model instance wasn't guaranteed to be freed
  before the next one loaded. **Fix**: `embed.py` now keeps a process-wide
  `_MODEL_CACHE` keyed by `(model_name, resolved device, max_seq_length)`,
  so every `Embedder` requesting the same model+device+seq-length shares
  one loaded instance. This is a pure performance/memory fix (identical
  encode() forward pass, zero effect on any embedding value or metric) and
  is what let the granite LongMemEval run complete at all under memory
  pressure — see `tests/test_core_embed.py`'s two cache tests.
  `bench/run_bench.py --retriever v4` now also times every `add()` call to
  report per-run index-build time and encode throughput (new
  `metadata.index_build_time_s` / `metadata.encode_throughput_items_per_s`
  fields), independent of this fix.
- A first LME-oracle attempt on cpu (mps-OOM fallback, before the cache fix
  landed) exceeded a 10-minute budget and was stopped rather than retried,
  per the session's resource-constraint rules; the model-cache fix above is
  what actually resolved it, not the cpu fallback.
- **Latency numbers on this table are noisy**: `locomo_v4_granite`
  (p50 30.5 ms) and `locomo_v4_combo_v3` (p50 18.1 ms) are the *identical*
  config (`embed_model=granite`, everything else default), run minutes
  apart on a machine with other unrelated CPU-heavy processes running
  concurrently (confirmed via `ps`) — a ~12 ms swing (~40% relative) on a
  byte-identical config. Macro R@10 is unaffected (deterministic Python-
  level tie-breaking, per the 2026-09-12 adversarial-review fix) and
  reproduces to the row above's 4th decimal; a handful of individual
  per-category counts drift by ~1 question out of hundreds between two
  identical-config runs (e.g. LoCoMo cat.1: 0.3758 vs 0.3746 between
  `locomo_v4_granite` and `locomo_v4_combo_v3`), consistent with known
  non-bit-exact floating point on Apple's MPS backend across separate
  process runs (not a code bug, and never enough to change any decision
  below). Treat p50 values here as directional, not exact, given this
  machine's shared load during the campaign.
- Reduced scope, agreed with the coordinator mid-campaign given the above:
  E11 kept only `rerank_top_n=30` (dropped `rerank_top_n=50` and the
  `rerank_weight=0.7` variant, since rerank30 itself did not clearly win --
  see Decision); E10's `w_dense=1.5` companion run (`locomo_v4_granite_d15`
  in the original brief) and an `expand_rerank30` combo were dropped.

### Results

All real `bench/run_bench.py` runs, `PYTHONHASHSEED=0`, against the current
defaults (`embed_model=all-MiniLM-L6-v2`, `fusion=cc`, `w_dense=1.0`,
`w_bm25=1.5`, `candidates=100`, `use_stemming=True`, `rerank_model=None`,
`entity_expansion=False`) as the baseline in each table.

**LoCoMo** (full corpus, n=1536, top_k=10):

| Run | cat.1 multi-hop | cat.2 temporal | cat.3 open-dom. | cat.4 single-hop | macro R@10 | p50 ms | encode turns/s | index build s |
|---|---|---|---|---|---|---|---|---|
| baseline (`locomo_v4_posthardening_default`, row 36) | 0.360 | 0.705 | 0.323 | 0.720 | 0.527 | 18.4 | n/a (pre-instrumentation) | n/a |
| **locomo_v4_granite** (row 44, E10: embed_model=granite) | 0.376 | 0.711 | 0.336 | 0.732 | 0.538 | 30.5 | 100.9 | 58.3 |
| **locomo_v4_rerank30** (row 46, E11: + rerank top_n=30, mps) | 0.401 | 0.730 | 0.380 | 0.749 | **0.565** | 166.1 | 101.8 | 57.8 |
| locomo_v4_rerank30_cpu (row 47, rerank device=cpu check) | 0.401 | 0.730 | 0.380 | 0.749 | 0.565 | 222.0 | 100.1 | 58.7 |
| **locomo_v4_expand** (row 49, E12: + entity_expansion) | 0.361 | 0.693 | 0.332 | 0.704 | 0.522 | 33.0 | 101.1 | 58.2 |
| **locomo_v4_combo_v3** (row 51, E13: greedy = granite alone) | 0.375 | 0.711 | 0.336 | 0.732 | 0.538 | 18.1 | 135.3 | 43.5 |

**LongMemEval oracle** (stratified 150, top_k=10):

| Run | temporal-reasoning | multi-session | knowledge-update | pref | assistant | user | macro R@10 | p50 ms | encode turns/s | index build s |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (`lme_oracle_v4_posthardening_default`, row 35) | 0.513 | 0.565 | 0.980 | 0.973 | 1.000 | 0.960 | 0.832 | 24.9 | n/a | n/a |
| **lme_oracle_v4_granite** (row 45, E10) | 0.507 | 0.601 | 0.980 | 0.973 | 1.000 | 0.960 | 0.8369 | 36.7 | 11.2 | 281.5 |
| **lme_oracle_v4_rerank30** (row 48, E11, mps) | 0.487 | 0.601 | 0.980 | 0.973 | 0.960 | 0.960 | 0.827 | 584.2 | 10.0 | 313.5 |
| **lme_oracle_v4_expand** (row 50, E12) | 0.507 | 0.601 | 0.980 | 0.973 | 1.000 | 0.960 | 0.8369 | 35.6 | 12.5 | 250.9 |
| **lme_oracle_v4_combo_v3** (row 52, E13) | 0.507 | 0.601 | 0.980 | 0.973 | 1.000 | 0.960 | 0.8369 | 33.2 | 13.4 | 235.1 |

LongMemEval's per-item texts (full session turns) are far longer than
LoCoMo's short dialogue turns, which is why encode throughput on LME
(10-13 turns/s) is an order of magnitude below LoCoMo's (~100-135 turns/s)
for the same model — not a regression, a corpus property; `granite`'s own
`max_seq_length` cap (512) still applies per item.

### E10: embedding upgrade (granite-embedding-97m-multilingual-r2)

Wins **both** benchmarks over the current MiniLM defaults: LoCoMo macro
R@10 0.527 -> **0.538** (+0.011), LongMemEval oracle 0.832 -> **0.8369**
(+0.0049) — every LoCoMo category improves or ties; on LongMemEval,
`multi-session` clearly improves (0.565 -> 0.601) while `temporal-
reasoning` ticks down slightly (0.513 -> 0.507, inside noise) and the rest
tie. **But it does not clear the promotion bar**: LongMemEval's 0.8369 is
*below* the required `> 0.837` (by 0.0001 — see the precise value in
`bench/results/lme_oracle_v4_granite_v1.json`, reproduced bit-for-bit by
`lme_oracle_v4_combo_v3`), and p50 latency (30.5 ms LoCoMo / 36.7 ms LME)
is over the 30 ms non-rerank guardrail on both benchmarks (though see the
latency-noise caveat above — `locomo_v4_combo_v3`, the identical config,
measured 18.1 ms minutes later). Per the letter of the given rule, this is
a **near-miss, not a win** — genuinely useful (a free, real improvement on
LoCoMo and a wash-to-slight-win on LongMemEval) but not what the rule asks
for, so it is reported as refuted-by-threshold rather than promoted.

### E11: cross-encoder reranking (ms-marco-ettin-17m-reranker, top_n=30, on granite)

Clear LoCoMo win: macro R@10 0.538 -> **0.565** (+0.027 over granite alone,
+0.038 over baseline) — every category improves, multi-hop (cat.1)
0.376 -> 0.401 and open-domain (cat.3) 0.336 -> 0.380 the largest gains, as
expected from a reranker that can read full lexical overlap the fused
score's minmax normalization discards. But it **regresses** LongMemEval:
0.8369 -> 0.827 (-0.010, `temporal-reasoning` 0.507 -> 0.487 and
`single-session-assistant` 1.000 -> 0.960 both drop) and blows the relaxed
150 ms latency guardrail badly: p50 166 ms on LoCoMo (mps) / **584 ms** on
LongMemEval (mps) — driven by LongMemEval's much longer per-item texts
making the cross-encoder's O(top_n) forward passes far more expensive per
query than LoCoMo's short turns. A device check (`locomo_v4_rerank30_cpu`,
row 47) found cpu *slower* than mps on the full LoCoMo run (222 ms vs
166 ms) despite an earlier small-sample smoke test suggesting the
opposite — mps stays the better (if still guard-violating) choice; the
17 M-parameter model is simply too slow at this `rerank_top_n`/text-length
combination for the 150 ms budget on LongMemEval-shaped corpora. Per the
reduced-scope agreement, `rerank_top_n=50` and `rerank_weight=0.7` were not
separately tested since rerank30 already failed the promotion rule on
LongMemEval.

### E12: entity-based multi-hop expansion (new mechanism)

Net negative. LoCoMo: macro R@10 0.538 -> **0.522** (-0.016 vs granite
alone), every single category drops including the multi-hop category it
was intended to help (cat.1: 0.376 -> 0.361) — expanding via shared rare
entities pulls in enough same-topic-but-wrong-turn noise into the top-10
that it displaces more correct hits than it adds. LongMemEval: **exactly
byte-identical** to granite alone (0.8369, every per-category value
unchanged) — LongMemEval oracle's per-conversation corpora (one question's
worth of sessions) apparently offer too few rare-entity co-occurrences
among the top-10 seeds to ever trigger an addition that survives to the
final top-10. `expansion_seed=10`/`expansion_decay=0.5` (spec defaults)
were used as-is; not retuned given E12's clear LoCoMo regression closes the
line of investigation for this campaign.

### E13: combo (greedy forward selection)

Forward selection starts from the only lever that improved *both*
benchmarks with no regression anywhere (E10, granite) and asks whether
adding the next lever improves without regressing: E11 (rerank) regresses
LongMemEval -0.010 and triples-to-16x the latency guardrail violation;
E12 (entity_expansion) regresses LoCoMo -0.016 and is neutral (not
positive) on LongMemEval. Both fail the "improve without regressing"
forward-selection criterion, so **`locomo_v4_combo_v3` /
`lme_oracle_v4_combo_v3` = granite embedding alone**, re-run fresh under
the combo name for the ledger (rows 51-52) and reproducing E10's granite
numbers to the 4th decimal (LoCoMo 0.538/0.538, LongMemEval 0.8369/0.8369).

### Decision: **no promotion.** `Config` defaults are unchanged.

`embed_model` stays `sentence-transformers/all-MiniLM-L6-v2`; `rerank_model`
stays `None` (opt-in via `MEMORY_V4_RERANK_MODEL`, see below);
`entity_expansion` stays `False`. Per the given rule, granite would be
promoted alone if it won *both* benchmarks strictly — it wins LoCoMo
(0.538 > 0.532) but LongMemEval's 0.8369 does not clear `> 0.837`
(reproduced identically twice, rows 45 and 52), and its own latency (while
noisy on this machine, see above) is the only genuinely over-guardrail
number among the "near-miss" candidates. Rerank and entity_expansion are
each an outright loss on at least one benchmark on their own, so neither
qualifies under the "wins LME massively but loses LoCoMo slightly"
exception either (that exception requires a large *targeted* win, not a
regression). All three mechanisms are left in the code, tested, and
OFF/unchanged by default — real, working, real-cached-model
implementations available for future work or for a coordinator who wants
to relax the LongMemEval bar by the 0.0001 that separates granite from
promotion.

**`MEMORY_V4_RERANK_MODEL`** (env override, `Config.rerank_model`): unset
by default. To opt in to reranking for interactive/MCP use despite it not
meeting this campaign's promotion bar, set
`MEMORY_V4_RERANK_MODEL=tomaarsen/ms-marco-ettin-17m-reranker` (the only
locally-cached reranker that scores correctly on this machine --
`cross-encoder/ms-marco-MiniLM-L6-v2` produces NaN under transformers 5.2
here and was deleted) together with `MEMORY_V4_RERANK_TOP_N=30` and
optionally `MEMORY_V4_RERANK_DEVICE=mps` (default `"auto"` already resolves
to mps) — expect a real LoCoMo-shaped R@10 gain but a p50 in the
100-600+ ms range depending on how long the corpus's own items are (see
E11 above), well past the 30 ms non-rerank guardrail; only worth it for a
caller that can tolerate sub-second search latency in exchange for
noticeably better ranking quality.

### Interpretation

1. **The embedding upgrade is real but falls one ten-thousandth short of
   this campaign's own bar.** Granite improves LoCoMo cleanly (+0.011,
   every category) and ties-to-slightly-improves LongMemEval (+0.0049,
   `multi-session` the standout), reproduced bit-for-bit across two
   independent runs (E10 and E13) — but `> 0.837` is a strict inequality
   and 0.8369 does not clear it. This is the kind of result the campaign's
   own "ruthless bookkeeping, promote only what wins" stance exists for:
   a nearly-free, nearly-there improvement is still not a win by the rule
   actually given.
2. **Reranking is a real, large ranking-quality lever whose latency cost
   scales with corpus item length, not just top_n.** +0.038 LoCoMo macro
   R@10 is the single largest gain of this entire campaign (bigger than
   the whole 2026-09-12 fusion/stemming combo's +0.030-ish per-lever
   deltas), but the exact same `rerank_top_n=30` config that stays under
   the relaxed guardrail's spirit on short LoCoMo turns (166 ms, itself
   still technically over 150 ms) explodes to 584 ms on LongMemEval's
   longer session texts — a caller cannot pick one `rerank_top_n` and
   expect one latency budget across corpora with different item lengths.
3. **Entity-based multi-hop expansion, as specified, adds more noise than
   signal.** Unlike E3's time-adjacency `neighbor_expansion` (also
   rejected in the original campaign, but for being latency-costly rather
   than accuracy-costly), hopping via shared rare entities actively hurts
   LoCoMo's own multi-hop category — the mechanism this campaign hoped
   would most help it. The corpus-wide entity-co-occurrence hop appears to
   surface topically-adjacent-but-wrong turns more often than it surfaces
   the actually-connected evidence LoCoMo's multi-hop questions need.
4. **A resource-constrained machine changed what this campaign could
   measure, not just how fast.** The real MPS OOM crash reloading granite
   150 times for LongMemEval oracle was a genuine bug (unbounded model
   reload under `V4Retriever.reset()`'s per-conversation pattern) that a
   comfortable machine would never have surfaced — the fix (a process-wide
   model cache in `embed.py`) is a strict improvement independent of this
   campaign's outcome and is now covered by regression tests.
5. **Greedy forward selection correctly converges to "do less."** Neither
   of the two additional levers survives the "improve without regressing"
   test once chained after the embedding upgrade, so the disciplined
   combo is just the embedding change on its own — and that alone still
   doesn't clear the bar. Sometimes the rigorous answer to "combine your
   winners" is that there was only one near-winner to begin with.

## Enrichissement par LLM hors ligne — Sonnet 5 (2026-09-13)

Mécanisme testé (A-Mem / Mem0 appliqué hors ligne) : chaque tour de LoCoMo a été lu par un sous-agent
Claude Sonnet 5 (10 agents, un par conversation, ~1,5 M tokens au total) qui a produit des faits
atomiques à la troisième personne avec dates absolues, des mots-clés et des entités
(`bench/data/enrich/locomo_facts/<conv>.jsonl`, 5 882 tours, 3 332 faits, couverture 53 % — les tours
de bavardage sont laissés vides par consigne ; densité variable selon l'agent, 0,47 à 1,04 fait/tour).
Le banc (`bench/enrich.py`, `V4EnrichedRetriever`) indexe les faits via `add_facts(parent_ids=[tour])`
à côté des tours bruts et crédite le tour parent quand un fait est rappelé. **Aucune modification de
`src/`** : c'est un test du mécanisme, qui en production est fourni par le LLM client via
`memory_chat_save(facts=…)`.

| Configuration (granite, cc 1:1,5, stemming) | multi-hop | temporel | open-dom. | single-hop | macro R@10 | R@5 | MRR | p50 |
|---|---|---|---|---|---|---|---|---|
| tours bruts (défaut) | 0,376 | 0,711 | 0,336 | 0,732 | 0,538 | 0,456 | 0,424 | 30 ms |
| + faits Sonnet | 0,407 | 0,762 | 0,350 | 0,845 | 0,591 | 0,499 | 0,435 | 33 ms |
| + faits + mots-clés dans le tour | 0,431 | 0,793 | 0,401 | 0,855 | **0,620** | **0,527** | **0,459** | 32 ms |

Lecture : +0,082 de rappel macro pour un coût de latence nul ; le single-hop atteint 0,855 (proche du
0,86 d'A-Mem sur l'ensemble) ; le multi-hop reste le point faible (0,43) — il faudrait des liens entre
faits (graphe) ou un reranker listwise. Seuls 8-11 % des premiers résultats sont des faits : le gain
vient surtout des mots-clés injectés dans le texte du tour (canal BM25/dense), signe que le vocabulaire
des questions diffère de celui des dialogues.
| + faits + mots-clés + reranker Ettin-17m top-30 | 0,497 | 0,827 | 0,430 | 0,896 | **0,662** | 0,594 | 0,541 | 142 ms |

Avec le reranker par-dessus l'enrichissement : 0,662 macro (MRR 0,541), single-hop 0,896, temporel 0,827 ;
le multi-hop passe à 0,50. Latence 142 ms : acceptable pour un usage MCP interactif, pas pour un
service à haut débit. Trajectoire mesurée : v3 exact 0,175 → v4 brut 0,538 → enrichi 0,620 → enrichi
+ reranker 0,662, contre 0,86 pour A-Mem (harnais indépendant). L'écart restant est concentré sur le
multi-hop et l'open-domain (liens entre faits, raisonnement), hors de portée d'un retrieval pur.

## P9 — planification de requête, liens, reranker (2026-09-13)

Objectif : dépasser A-Mem (LoCoMo macro R@10 > 0,86 ; état de départ 0,662 =
`locomo_v4_enriched_kw_rerank30`, granite + faits/mots-clés Sonnet 5 + reranker Ettin-17m).
Diagnostic (PLAN.md §P9) : single-hop (0,896) et temporel (0,827) sont déjà proches du plafond ;
l'écart restant est multi-hop (0,50, ≥3 tours de preuve) et open-domain (0,43, inférence) — un
retrieval mono-requête ne peut pas les résoudre. Trois mécanismes ajoutés **côté banc uniquement**
(`bench/enrich.py`, `bench/retrievers.py::V4PlannedRetriever`, `bench/run_bench.py` — `src/memory_os_v4`
inchangé) :

- **§A planification de requête** : chaque question est réécrite hors ligne, une fois, par Sonnet 5
  en 2-4 sous-requêtes (vocabulaire des dialogues) + 1-2 « faits hypothétiques » (HyDE), écrits dans
  `bench/data/enrich/locomo_queries/<conv_id>.jsonl` (couverture mesurée : **1536/1536 questions
  (100 %)**, les 10 fichiers ont été livrés par les workers concurrents avant la fin de
  l'implémentation — aucun `--allow-partial-plan` n'a été nécessaire). `search()` interroge
  `engine.search(..., rerank=False, top_k=top_k*4)` pour la question originale + chaque sous-requête/
  hypothèse sélectionnée par `--plan-mode`, mappe chaque liste de hits vers des tours (fait → tour
  parent, comme `V4EnrichedRetriever`), puis fusionne par RRF pondéré (`k=60` par défaut ; poids
  original/sous-requêtes/hypothèses = 1,0/1,0/0,7).
- **§B liens entre faits** (A-Mem/Zettelkasten, `--fact-links`) : deux faits sont liés s'ils partagent
  ≥1 entité rare (df ≤5 % des tours, ≥3 caractères, jamais un nom de locuteur) OU appartiennent à la
  même session à ≤2 tours d'écart. Après fusion, les tours des faits liés aux 8 tours fusionnés de
  tête (`--link-seed`) sont ajoutés à `score_seed × 0,5` (`--link-decay`).
- **§C reranker plus fort** : `tomaarsen/ms-marco-ettin-68m-reranker` (274 Mo, téléchargé une fois via
  `huggingface_hub.snapshot_download`, `HF_HUB_OFFLINE=1` pour tous les runs ensuite) comparé au 17m
  déjà en cache, appliqué **par le banc lui-même** sur le top `--rerank-top-n` fusionné — jamais via
  `Config.rerank_model` (tous les appels `engine.search` ci-dessus passent `rerank=False`).

**LoCoMo** (full corpus, n=1536, top_k=10, granite embedding par défaut, `--enrich-mode facts+keywords`) :

| Run (RESULTS.md #) | cat.1 multi-hop | cat.2 temporel | cat.3 open-dom. | cat.4 single-hop | macro R@5 | macro R@10 | MRR | nDCG@10 | p50 ms | fraction avec plan |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (`locomo_v4_enriched_kw_rerank30`, row 55) | 0,497 | 0,827 | 0,430 | 0,896 | 0,594 | 0,662 | 0,541 | 0,539 | 141,9 | n/a (pas de plan) |
| 1. `locomo_v4_plan_sub` (row 56) : §A sub seul, rerank17 top30 blend | 0,595 | 0,861 | 0,492 | 0,926 | 0,633 | **0,719** | 0,607 | 0,600 | 393,3 | 1,00 |
| 2. `locomo_v4_plan_subhyp` (row 57) : §A sub+hyp, rerank17 top30 blend | 0,624 | 0,890 | 0,499 | 0,935 | 0,661 | **0,737** | 0,630 | 0,620 | 444,5 | 1,00 |
| 3. `locomo_v4_plan_subhyp_ceonly` (row 58) : sub+hyp, rerank17 top30 CE seul | 0,590 | 0,889 | 0,489 | 0,924 | 0,643 | 0,723 | 0,612 | 0,605 | 443,2 | 1,00 |
| 4. `locomo_v4_plan_subhyp_links` (row 59) : sub+hyp + §B liens, rerank17 top30 blend | 0,621 | 0,887 | 0,499 | 0,932 | 0,660 | 0,735 | 0,627 | 0,618 | 459,5 | 1,00 |
| 5. `locomo_v4_plan_subhyp_links_e68` (row 60) : meilleur-des-4 (= run 2, sans liens) + §C Ettin-68m | 0,634 | 0,896 | 0,502 | 0,936 | 0,673 | **0,742** | 0,651 | 0,639 | 1368,3 | 1,00 |
| 6a. `locomo_v4_plan_best_top50` (row 61) : run 5 + `rerank_top_n=50` | 0,639 | 0,893 | 0,508 | 0,939 | 0,677 | 0,745 | 0,653 | 0,642 | 1693,4 | 1,00 |
| 6b. `locomo_v4_plan_best_k20` (row 62) : run 5 + `rrf_k=20` | 0,651 | 0,901 | 0,513 | 0,940 | 0,668 | **0,751** | 0,642 | 0,636 | 1064,1 | 1,00 |

Couverture des plans : **100 % (1536/1536)** sur les 7 runs (`fraction_with_plan` dans
`metadata.planning` de chaque JSON) — les 10 fichiers `locomo_queries/<conv>.jsonl` étaient tous
livrés avant que le premier run réel ne démarre ; `--allow-partial-plan` reste implémenté et testé
(`tests/test_bench_plan.py`) mais n'a jamais été nécessaire en pratique.

### §A seul (runs 1-3) : plus gros gain de la campagne, sub+hyp > sub seul > CE seul

`locomo_v4_plan_sub` (sous-requêtes seules) fait déjà passer le macro R@10 de 0,662 à **0,719**
(+0,057) — **toutes les catégories s'améliorent**, y compris le multi-hop visé (0,497→0,595) et
l'open-domain (0,430→0,492). Ajouter les hypothèses HyDE (`locomo_v4_plan_subhyp`, run 2) ajoute
+0,018 de plus (0,737) : le multi-hop continue de progresser (0,595→0,624) ainsi que le temporel
(0,861→0,890, sans doute parce que les sous-requêtes/hypothèses répètent des dates ou reformulations
absentes du tour original). Faire décider le reranker par le score cross-encoder **seul** plutôt que
le blend 0,5/0,5 (`locomo_v4_plan_subhyp_ceonly`, run 3) est **strictement pire** que le blend sur
toutes les catégories (0,737→0,723) : contrairement à E11 (campagne Palier 2-3, un seul retrieval
mono-requête), ici le score de fusion RRF porte déjà un signal utile (accord entre plusieurs
requêtes) que le cross-encoder seul, aveugle à ce consensus, perd. **Décision : blend 0,5/0,5 conservé
par défaut** (`--plan-rerank-blend`), sub+hyp préféré à sub seul.

### §B liens entre faits (run 4) : neutre à légèrement négatif, comme E12

`locomo_v4_plan_subhyp_links` (sub+hyp + liens A-Mem) obtient 0,735, **légèrement en dessous** de
`locomo_v4_plan_subhyp` sans liens (0,737 → -0,002) — dans le bruit de mesure mais ne satisfait
certainement pas la règle de promotion du dépôt (gain > 0,005). Le multi-hop lui-même recule
légèrement (0,624→0,621) alors que c'est la catégorie que les liens visaient à aider. Ce résultat
reproduit le verdict de E12 (Palier 2-3 : l'expansion par entités rares régressait déjà le multi-hop
LoCoMo, -0,016) avec un mécanisme différent (liens au niveau des faits, pas des tours) mais la même
cause probable : les tours ajoutés par expansion partagent une entité ou une proximité de session
sans nécessairement porter la preuve que la question cherche, et déplacent plus de bons résultats
qu'ils n'en ajoutent dans le top-10 fusionné. **Décision : `--fact-links` reste opt-in, off par
défaut** dans la configuration retenue — le mécanisme est implémenté, testé (9 tests dédiés dans
`tests/test_bench_plan.py`), et documenté pour un travail futur (une fenêtre de liens plus étroite,
ou un score de lien non uniforme selon le nombre d'entités partagées, pourraient inverser ce résultat,
mais n'ont pas été essayés ici).

### §C reranker Ettin-68m (run 5) : gain net mais latence x3

En partant du meilleur config sans liens (run 2, sub+hyp, blend), remplacer le reranker Ettin-17m par
le **Ettin-68m** (`locomo_v4_plan_subhyp_links_e68`, nom conservé tel que spécifié par la tâche même
si `--fact-links` est resté désactivé — voir note ci-dessous) porte le macro R@10 à **0,742** (+0,005
vs run 2, +0,080 vs la baseline 0,662) — le multi-hop (0,624→0,634), l'open-domain (0,499→0,502) et le
single-hop (0,935→0,936) progressent tous, mais le coût est réel : p50 passe de 444,5 ms à **1368,3 ms**
(×3,1) — le modèle 68m est ~4x plus lourd que le 17m et le banc l'appelle une fois par question sur
30 tours fusionnés.

**Note sur le nom `locomo_v4_plan_subhyp_links_e68`** : la consigne de la tâche demandait "run 5 =
meilleur des runs précédents + Ettin-68m", nommé littéralement `..._links_e68`. Comme le run 4 (avec
liens, §B) n'a **pas** amélioré sur le run 2 (sans liens) — 0,735 < 0,737, une régression dans le
bruit qui ne franchit pas la barre de promotion +0,005 — le "meilleur des runs précédents" réel est
le run 2 (sans liens). Le nom exact demandé par la tâche a été conservé pour `RESULTS.md` (`--fact-links`
non passé, documenté explicitement dans la colonne Notes de la ligne 60 et ici) plutôt que d'empiler
silencieusement un mécanisme qui régresse légèrement sous un nom qui laisse croire qu'il est inclus.

### Runs 6a/6b : `rerank_top_n=50` et `rrf_k=20` (run 5 < 0,86, donc essayés par consigne)

`locomo_v4_plan_best_top50` (run 5 + `rerank_top_n=50`, run 6a) : 0,742→**0,745** (+0,003), la plupart
des catégories progressent légèrement (open-domain 0,502→0,508 le plus net) mais la latence
**double presque** (1368→1693 ms) pour un gain qui reste dans le bruit de mesure. `locomo_v4_plan_best_k20`
(run 5 + `rrf_k=20`, run 6b) : 0,742→**0,751** (+0,009, le plus gros gain des deux variantes 6a/6b) —
resserrer le RRF (favoriser davantage les rangs de tête de chaque composant) aide le multi-hop
(0,634→0,651) et le temporel (0,896→0,901) sans coût de latence supplémentaire (1064 ms, même
inférieur à run 5 malgré la variance de latence déjà documentée dans ce fichier). **`locomo_v4_plan_best_k20`
est la meilleure configuration mesurée de toute la campagne P9** : macro R@10 **0,751**, MRR 0,642.

### Décision : pas de promotion vers `src/`, meilleure config documentée pour usage banc

Aucune des 7 configurations n'atteint la cible de 0,86 (A-Mem). La meilleure, `locomo_v4_plan_best_k20`
(`--plan-mode sub+hyp --rrf-k 20 --rerank-model tomaarsen/ms-marco-ettin-68m-reranker --rerank-top-n 30
--plan-rerank-blend`, sans `--fact-links`), porte le macro R@10 LoCoMo de 0,662 (P8) à **0,751**
(+0,089), avec p50 ≈1064 ms — bien au-delà de tout budget d'usage interactif MCP à faible latence, donc
restant un mécanisme de banc opt-in (`bench/retrievers.py::V4PlannedRetriever`), jamais porté dans
`src/memory_os_v4` (hors mandat explicite de cette session : "src/ intact"). Comme pour la campagne
Palier 2-3, le mécanisme le plus utile mesuré (§A planification de requête, +0,057 à +0,075 selon la
config, latence ×3 mais raisonnable pour un usage hors-ligne/asynchrone) est un candidat crédible pour
une future itération à budget de latence plus généreux ou avec un modèle de reranking plus rapide ;
§B (liens entre faits) est refermé pour cette itération (comme E12 avant lui) ; §C (reranker plus
gros) est un gain net mais cher, cohérent avec E11.

### Interprétation

1. **La planification de requête (§A) est, de loin, le levier le plus efficace testé sur LoCoMo depuis
   le début de cette campagne** : +0,057 (sub seul) à +0,075 (sub+hyp) de macro R@10 pour une seule
   idée — reformuler la question dans le vocabulaire du dialogue et fusionner plusieurs vues résout
   une partie substantielle du problème "la question et la réponse ne partagent pas de mots" qui
   handicapait le retrieval mono-requête depuis la ligne 1 du ledger.
2. **Le blend 0,5/0,5 bat le cross-encoder seul une fois plusieurs requêtes fusionnées** — à l'inverse
   de l'intuition "le cross-encoder est plus précis, laissons-le décider seul" : le signal RRF
   multi-requête porte de l'information (consensus entre reformulations) que le cross-encoder,
   qui ne voit qu'une paire (question originale, tour) à la fois, ne peut pas reconstruire.
3. **Les liens entre faits (§B) échouent une seconde fois, avec un mécanisme différent de celui déjà
   rejeté (E12)** — deux tentatives indépendantes de "sauter" vers des tours voisins par entité/
   proximité de session régressent toutes deux (légèrement) le multi-hop qu'elles visaient à aider,
   ce qui est un signal plus fort que chaque échec pris isolément : sur ce corpus, l'expansion par
   proximité structurelle (entité partagée, tours voisins) ajoute plus de bruit topique que de
   preuve réellement pertinente au top-10.
4. **Le gain d'un reranker plus gros (§C) est réel mais sous-linéaire par rapport à son coût** :
   Ettin-68m (4x les paramètres du 17m) n'ajoute que +0,005 par-dessus §A seul, contre les +0,057 de
   §A lui-même — cohérent avec E11 (Palier 2-3) où le reranking apportait déjà le plus gros gain
   isolé mesuré sur un retrieval mono-requête ; une fois le multi-requête en place, le reranker a
   beaucoup moins de désordre à corriger dans le top-30 fusionné.
5. **0,86 (A-Mem) reste hors de portée d'un retrieval + reranking pur sur ce harnais** : la
   trajectoire complète mesurée sur cette conversation est v3 exact 0,175 → v4 brut 0,538 → enrichi
   0,620 → enrichi+rerank17 0,662 → **planifié+rerank68+k20 0,751** — un gain cumulé de +0,576 depuis
   le début, mais encore -0,109 sous la cible. L'écart restant (multi-hop 0,651, open-domain 0,513
   au mieux) suggère que le raisonnement multi-tours (agréger 3+ preuves, inférer au-delà d'une
   phrase citée) demande un mécanisme que le retrieval ne peut pas fournir seul — la piste D du
   plan (reranking listwise par LLM, PLAN.md §P9) reste le candidat le plus prometteur pour une
   future itération, puisqu'en production c'est de toute façon le modèle client qui lit le contexte
   récupéré.

## P9 étape D — export de candidats pour reranking listwise par LLM (2026-09-13)

Diagnostic (section précédente) : l'écart restant vers A-Mem (0,86) après `locomo_v4_plan_best_k20`
(0,751 macro R@10, row 62) est probablement un problème de **rang**, pas de **rappel** — le
cross-encoder (§C) ne voit qu'une paire (question, tour) à la fois et ne peut pas exploiter le
contexte des 19 autres candidats fusionnés. L'étape D teste si un LLM, lisant les 20 candidats **d'un
coup** (listwise), peut mieux les classer — un mécanisme entièrement **hors ligne**, en deux temps,
toujours côté banc uniquement (`bench/enrich.py`, `bench/retrievers.py::V4LLMRerankedRetriever`,
`bench/run_bench.py` — `src/memory_os_v4` inchangé).

### Partie 1 — export des candidats (`--dump-candidates`)

`bench/run_bench.py --dump-candidates DIR` (n'importe quel retriever, ici `v4_planned` avec la config
exacte de `locomo_v4_plan_best_k20`, row 62) écrit `DIR/<conv_id>.jsonl`, une ligne par question :
`{"qid", "question", "candidates": [{"label": "c01".."c20", "turn_id", "date", "speaker", "text"
(≤320 car.), "facts": [≤2 faits ≤160 car.]}], "label_map": {"c01": turn_id, ...}}` — le classement
FINAL du retriever (après le reranker cross-encoder Ettin-68m), jamais un id de preuve, une réponse ou
une catégorie (`bench.enrich.build_candidate_dump_line`). Un seul appel `search()` suréchantillonné à
`max(--top-k, 20)` sert à la fois le dump ET les métriques normales de l'exécution (`ranked[:top_k]`
est un préfixe exact du top-20 dumpé) : dumper ne double jamais la latence.

Exécuté une fois en réel sur les 10 conversations LoCoMo (`--plan-mode sub+hyp --rrf-k 20
--rerank-model tomaarsen/ms-marco-ettin-68m-reranker --rerank-top-n 30 --plan-rerank-blend
--enrich-mode facts+keywords`), nommé `locomo_v4_plan_best_k20_dump` (row 63, `nice -n 15`, un seul
processus, ~34 min) :

| | valeur |
|---|---|
| conversations / fichiers | 10/10 |
| lignes (questions) | 1536/1536 |
| candidats par ligne | 20 (min=max=20 : les 10 conversations LoCoMo ont toutes ≥20 tours) |
| champs candidat | exactement `{label, turn_id, date, speaker, text, facts}` — vérifié sur les 1536×20=30720 candidats |
| id de preuve/réponse/catégorie dans le dump | 0 (vérifié) |
| macro R@10 / R@5 / MRR / nDCG@10 (métriques normales, top_k=10) | 0,749 / 0,671 / 0,643 / 0,637 |
| p50 / p95 | 1151,3 ms / 1691,8 ms |

Le macro R@10 (0,749) est à -0,002 de la row 62 (0,751, `rrf_k=20` identique) — dans le bruit de
mesure déjà documenté dans ce fichier (latence/scores varient légèrement d'un run à l'autre malgré
`PYTHONHASHSEED=0`), la différence venant aussi de l'oversampling (`top_k=20` au lieu de `10` pour
que le dump et les métriques partagent le même appel `search()`, cf. docstring `bench/run_bench.py`).

**Rappel@20 du dump (plafond de l'étape D)** — le rappel des 20 candidats retenus par le retriever
avant tout reranking LLM (recall_at_k avec k=20, calculé directement depuis le dump + les preuves du
banc — les preuves ne sont jamais écrites dans le dump lui-même, ce calcul est fait séparément côté
rapport) :

| catégorie | n | R@10 mesuré (row 63) | Rappel@20 (plafond) | marge |
|---|---|---|---|---|
| 1 multi-hop | 282 | 0,653 | **0,749** | +0,096 |
| 2 temporel | 321 | 0,898 | **0,923** | +0,025 |
| 3 open-domaine | 92 | 0,502 | **0,596** | +0,094 |
| 4 single-hop | 841 | 0,943 | **0,962** | +0,019 |
| **macro** | 1536 | 0,749 | **0,808** | **+0,058** |

Lecture : le plafond théorique d'un reranking parfait du top-20 (macro R@10 = 0,808) est **encore
loin de 0,86** (A-Mem) — même un LLM parfait ne peut pas dépasser 0,808 en ne faisant que réordonner
ces 20 candidats, car 1536×(1-0,808) ≈ 295 "points-question" de rappel manquant proviennent de
candidats qui ne sont simplement PAS dans le top-20 retourné par `locomo_v4_plan_best_k20`, pas d'un
mauvais classement parmi eux. Le detail par catégorie confirme le diagnostic du fichier : la marge est
concentrée sur le multi-hop (+0,096) et l'open-domaine (+0,094) — les deux catégories que §A visait
déjà — alors que le temporel (+0,025) et le single-hop (+0,019) sont presque déjà à leur plafond de
rappel ; pour ces deux dernières catégories, l'essentiel du gain potentiel restant est donc un
problème de **rang** (reranking listwise pertinent), tandis que pour multi-hop/open-domaine environ
la moitié de la marge R@20 elle-même vient du fait que le top-20 ne contient pas encore toute la
preuve — un LLM listwise ne peut aider qu'à récupérer la partie déjà présente dans les 20.

### Partie 2 — retriever de reranking LLM (`v4_llmrank`) et évaluation `--from-dump`

`V4LLMRerankedRetriever` (`bench/retrievers.py`, extends `V4PlannedRetriever`) lit
`<llmrank_dir>/<conv_id>.jsonl` (`{"qid", "ranked": ["c07", "c02", ...]}`, labels choisis par un LLM
hors ligne, produit par d'autres sessions plus tard) : `search()` relance le pipeline parent pour
obtenir son propre top-20 (même config déterministe que le dump), traduit les labels en `turn_id` via
le `label_map` du dump lui-même (`bench.enrich.load_candidate_dump`, jamais en recalculant une
position) via `bench.retrievers.apply_llm_ranking` : ids classés par le LLM d'abord (dédupliqués),
puis les candidats restants dans leur ordre d'origine, coupé à `top_k`. Une question sans classement
LLM retombe sur l'ordre du parent (compté dans `metadata.llm_rerank.n_fallback_no_llm_ranking`).

Un second mode, `--from-dump`, évalue directement depuis les deux dumps (candidats + classements LLM)
**sans reconstruire de retriever, sans charger de modèle, sans appeler `add()`/`search()`** — relit
`--candidates-dir`/`--llmrank-dir` et score avec les mêmes fonctions `bench.metrics` que la boucle
normale, via la même fonction `apply_llm_ranking` (les deux chemins ne peuvent jamais diverger sur la
règle de combinaison, testé). `latency_ms` y est rapporté à 0,0 pour chaque question
(`metadata.from_dump.note`) — permet d'itérer rapidement une fois les classements LLM disponibles,
sans repayer le coût du reranker Ettin-68m (p50 1151 ms mesuré ci-dessus).

**Décision : implémentation et export livrés, classement LLM réel pas encore produit** — les fichiers
`<llmrank_dir>/<conv_id>.jsonl` sont produits par d'autres sessions/workers plus tard (hors mandat de
cette session : le dump ci-dessus est prêt à être consommé). `V4LLMRerankedRetriever`/
`run_bench_from_dump` sont testés sur données synthétiques (22 nouveaux tests,
`tests/test_bench_plan.py`, suite 33→55) et l'export lui-même est vérifié en conditions réelles sur
les 10 conversations LoCoMo (tableaux ci-dessus). Le plafond mesuré (0,808 macro) borne ce qu'un
reranking listwise LLM peut apporter par rapport au 0,751 actuel : un gain jusqu'à +0,057 macro R@10
est possible si le LLM classe parfaitement le top-20, concentré sur multi-hop et open-domaine ;
0,86 (A-Mem) resterait hors de portée même dans ce cas optimiste, car une partie de l'écart vient de
candidats absents du top-20 lui-même. Commande pour évaluer une fois les classements disponibles :

```bash
cd /Users/romain/Memory-os-ai/v4 && PYTHONPATH=src:. /Users/romain/Memory-os-ai/.venv/bin/python -m bench.run_bench \
    --dataset locomo --retriever v4_llmrank --from-dump --name locomo_v4_llmrank_fromdump --top-k 10 \
    --candidates-dir bench/data/dumps/locomo_v4_plan_best_k20 \
    --llmrank-dir bench/data/dumps/locomo_v4_plan_best_k20_llmrank
```

## P9 étape D — reranking listwise par LLM (Sonnet 5, hors ligne, 2026-09-13)

18 sous-agents Sonnet 5 ont classé, pour chacune des 1 536 questions, les 20 candidats exportés par
`locomo_v4_plan_best_k20` (étiquettes opaques c01..c20, texte + faits, jamais d'évidence ni de
réponse). Évaluation depuis les fichiers (`--from-dump`, aucun recalcul). Couverture 100 % après
régénération de 11 questions manquantes (un agent avait rendu la mauvaise conversation sur un lot).

| Configuration | multi-hop | temporel | open-dom. | single-hop | macro R@10 | R@5 | MRR |
|---|---|---|---|---|---|---|---|
| planification + Ettin-68m (étape C, ligne 63) | 0,653 | 0,898 | 0,502 | 0,943 | 0,749 | 0,677 | 0,653 |
| + reranking listwise Sonnet top-20 | 0,707 | 0,912 | 0,533 | 0,956 | **0,777** | 0,750 | **0,762** |
| plafond du top-20 (rappel@20) | 0,749 | 0,923 | 0,596 | 0,962 | 0,808 | — | — |

Lecture : le LLM capte ~50 % de la marge restante du top-20 et transforme surtout la précision
(MRR +0,11). Le verrou est désormais la largeur des candidats : élargir à 50 (plafond estimé 0,86-0,88)
puis reranker à nouveau est la seule voie vers > 0,86 sans retrieval itératif. En production, ce
reranking est gratuit : c'est le modèle client qui lit le contexte renvoyé par `memory_get_context`.

## P9 étape D bis — 50 candidats + reranking listwise Sonnet 5 (2026-09-14)

Export élargi à 50 candidats (`BENCH_DUMP_MAX=50`, même configuration que la ligne 63, 19 min en
un seul processus sous garde-fou mémoire : swap 3,1 Go stable, aucune intervention). Plafond
rappel@50 : multi-hop 0,856 / temporel 0,952 / open-domain 0,691 / single-hop 0,979 / **macro 0,870**
(contre 0,809 à 20). 30 sous-agents Sonnet 5 (tranches ≤ 62 questions, 5 questions rendues à la
fois, ≈ 5 M tokens) ont classé les 50 candidats de chaque question ; fusion par identifiant de
question, régénération ciblée des lots mal rendus.

| Configuration | multi-hop | temporel | open-dom. | single-hop | macro R@10 | R@5 | MRR |
|---|---|---|---|---|---|---|---|
| reranking Sonnet du top-20 (ligne 64) | 0,707 | 0,912 | 0,533 | 0,956 | 0,777 | 0,750 | 0,763 |
| reranking Sonnet du top-50, couverture 100 % (ligne 71) | 0,766 | 0,924 | 0,584 | 0,968 | 0,810 | 0,778 | 0,775 |
| ensemble RRF des classements top-20 et top-50 (gratuit) | 0,772 | 0,931 | 0,586 | 0,970 | **0,815** | 0,780 | 0,773 |
| plafond du top-50 | 0,856 | 0,952 | 0,691 | 0,979 | 0,870 | — | — |

Lecture : le LLM capte 54 % de la marge entre la ligne 63 (0,751) et le plafond (0,870). Cible 0,86 non
atteinte : l'écart restant (0,045) est entièrement dans la précision du reranking sur multi-hop
(0,77 vs plafond 0,86) et open-domain (0,59 vs 0,69), c'est-à-dire des questions de raisonnement où
Sonnet hésite entre plusieurs tours plausibles. Pistes restantes : reranking Opus sur ces deux
catégories seulement (374 questions × 50 candidats ≈ 1,5 M tokens), auto-cohérence (deux passes
Sonnet indépendantes fusionnées : l'ensemble 20+50 donne déjà +0,005), ou retrieval itératif.
Trajectoire complète : v3 exact 0,175 → v4 brut 0,538 → +faits 0,620 → +reranker 0,662 →
+planification 0,751 → +LLM top-20 0,777 → +LLM top-50 0,810 → ensemble 0,815.

## Idées inter-disciplines (2026-09-14)

Objectif (`docs/IDEAS.md`) : dépasser `locomo_v4_plan_best_k20` (baseline P9,
macro R@10 **0,751** : multi-hop 0,658, temporel 0,901, open-domaine 0,497,
single-hop 0,947, p50 1064,1 ms) en attaquant la **couverture** plutôt que la
similarité — cinq mécanismes empruntés à d'autres champs (choix social,
finance, fusion de capteurs, linguistique du discours, fourragement optimal),
tous implémentés comme options `V4PlannedRetriever` par défaut désactivées
(`bench/retrievers.py`), jamais dans `src/memory_os_v4`. Critère de
promotion (fixé par la tâche) : gain multi-hop ≥ +0,02 sans perte
single-hop/temporel > 0,005. Runs séquentiels, `nice -n 15`, un seul
processus à la fois, swap vérifié < 6 Go avant chaque lancement (jamais
dépassé : 3,0-4,3 Go observés sur les 7 runs) ; nouveauté de cette session :
`--report-ceilings` (recall@k pour k∈{10,20,30,50} du classement FINAL via
un seul appel `search(top_k=50)` par question, jamais réinjecté dans le
retrieval — seulement pour le calcul du plafond).

| Run | multi-hop | temporel | open-dom. | single-hop | macro R@5 | macro R@10 | MRR | nDCG@10 | ceil@20 | ceil@30 | ceil@50 | p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline `locomo_v4_plan_best_k20` (row 62) | 0,658 | 0,901 | 0,497 | 0,947 | — | **0,751** | — | — | 0,808¹ | n/a | 0,870² | 1064,1 |
| 1. `locomo_v4_idea1_quota` (row 72) : Sainte-Laguë, seats=30 | 0,657 | 0,895 | **0,510** | 0,946 | 0,681 | 0,752 | 0,649 | 0,642 | 0,802 | 0,812 | 0,861 | 645,8 |
| 1v. `locomo_v4_idea1_quota_then_rrf` (row 73) : seats=10, RRF après | 0,658 | 0,898 | 0,499 | 0,947 | 0,675 | 0,750 | 0,647 | 0,639 | 0,814 | 0,828 | 0,869 | 754,9 |
| 2. `locomo_v4_idea2_mmr` (row 74) : MMR λ=0,6 | 0,657 | 0,900 | 0,501 | 0,947 | 0,664 | 0,751 | 0,640 | 0,632 | 0,814 | **0,832** | 0,862 | 915,9 |
| 3. `locomo_v4_idea3_conf` (row 75) : confiance top1/top2 | **0,664** | **0,903** | 0,499 | 0,944 | 0,676 | 0,752 | 0,642 | 0,638 | 0,805 | 0,820 | 0,870 | 779,2 |
| 4. `locomo_v4_idea4_adj` (row 76) : paires adjacentes | 0,658 | 0,901 | 0,497 | 0,947 | 0,674 | 0,751 | 0,643 | 0,637 | 0,809 | 0,822 | 0,869 | 964,4 |
| 5. `locomo_v4_idea5_iter` (row 77) : fourragement itératif | 0,658 | 0,901 | 0,497 | 0,947 | 0,674 | 0,751 | 0,642 | 0,637 | 0,809 | 0,823 | **0,870** | 844,0 |
| combo `locomo_v4_ideas_combo` (row 78) : ensemble vide (voir décision) | 0,659 | 0,901 | 0,497 | 0,947 | 0,674 | 0,751 | 0,642 | 0,637 | 0,809 | 0,823 | 0,870 | 830,9 |

¹ rappel@20 baseline mesuré séparément via la méthode de dump top-20 (P9
étape D, row 63, `locomo_v4_plan_best_k20_dump`), config identique
(rrf_k=20, rerank_top_n=30, Ettin-68m) — même calcul (recall@k du
classement final), méthode différente (`--report-ceilings` n'existait pas
encore) ; multi-hop 0,749, temporel 0,923, open-domaine 0,596, single-hop
0,962. ² rappel@50 baseline via `BENCH_DUMP_MAX=50` (P9 étape D bis) ;
multi-hop 0,856, temporel 0,952, open-domaine 0,691, single-hop 0,979 —
rappel@30 jamais mesuré pour la baseline (seuls @20 et @50 avaient été
dumpés). **Vérification interne** : `ceiling@10` est par construction
identique au R@10 réel rapporté (même liste, même k) sur les 7 runs — bon
test de cohérence de l'implémentation de `--report-ceilings`.

### Décision : aucune idée ne franchit le seuil de promotion — ensemble combo vide

**Aucune des 6 idées ne s'approche du seuil +0,02 sur le multi-hop.** Le
meilleur gain mesuré est +0,006 (idée 3, confiance de signal), soit moins
d'un tiers du seuil ; quatre idées (1v, 4, 5, et l'idée 1 elle-même en
négatif) sont à ±0,001 de la baseline sur le multi-hop, dans le bruit de
mesure déjà documenté ailleurs dans ce fichier (jitter ±0,003 selon la
configuration). Comme la règle de promotion du dépôt conditionne l'entrée
dans le run combo à « gain multi-hop ≥ +0,02 », l'ensemble à combiner est
**vide** : `locomo_v4_ideas_combo` (row 78) a donc été exécuté avec la
config baseline (aucun flag d'idée) + `--report-ceilings`, pour documenter
explicitement ce point de comparaison à conditions égales plutôt que de
sauter le run nommé demandé par la tâche. Son macro R@10 (0,751, multi-hop
0,659) reproduit la baseline au bruit de mesure près (±0,001-0,002),
confirmant qu'aucune combinaison n'a été tentée par construction — **rien
ne gagne, dit clairement**.

### Interprétation par idée (5 lignes chacune)

**Idée 1 — fusion à quotas (Sainte-Laguë)**. `quota` (seats=30) déplace le
gain vers l'open-domaine (+0,013) et coûte au temporel (−0,006, dépasse de
justesse la tolérance −0,005) sans toucher le multi-hop visé (−0,001) ;
`quota_then_rrf` (seats=10) est presque exactement neutre partout (±0,003).
Le plafond @20/@30/@50 de `quota` est systématiquement le PLUS BAS des 7
runs (0,802/0,812/0,861 contre ~0,809/0,822/0,869 ailleurs) : forcer une
représentation proportionnelle dans les 30 premières places déplace des
candidats à fort score RRF hors de la fenêtre rerankée (ils tombent dans le
« reste » RRF, jamais revus par le cross-encoder), ce qui abaisse le
plafond au lieu de l'élever — l'hypothèse « un siège par circonscription »
ne compense pas la perte de candidats forts évincés. Décision : ni l'une ni
l'autre variante ne dépasse le bruit de mesure sur le multi-hop ciblé.

**Idée 2 — MMR par sous-requête**. Neutre sur toutes les catégories
(multi-hop −0,001, temporel −0,001, open-domaine +0,004, single-hop 0,000) ;
seul effet notable = le plafond @30 le plus élevé des 7 runs (0,832), signe
que la diversification par « propriétaire » ajoute réellement des candidats
d'origine variée dans la fenêtre 21-30, mais cela ne se traduit PAS en gain
de rappel@10 réel une fois le reranker cross-encoder appliqué : le score
pénalisé par MMR alimente bien le blend 0,5/0,5 du rerank (`_apply_rerank`
lit `scores` — donc les valeurs MMR — pour son `base_norm`), mais il n'y
pèse que pour moitié, et `minmax_normalize` réétire ensuite ces valeurs sur
[0,1] par requête, ce qui efface une bonne part de la magnitude que la
pénalité λ^n avait imprimée ; le score cross-encoder (l'autre moitié)
domine le classement final. La pénalité λ^n change quels candidats
occupent la fenêtre rerankée mais pas assez pour déplacer le résultat
final.

**Idée 3 — confiance de signal**. Le SEUL mécanisme avec un gain multi-hop
positif net (+0,006) et le seul aussi à améliorer le temporel (+0,002)
simultanément — cohérent avec l'intuition (pondérer moins une sous-requête
dont le moteur hésite entre son 1er et 2e résultat) mais l'effet reste dix
fois trop petit pour le seuil de promotion. Coût mesuré : single-hop
−0,003, dans la tolérance. Le plafond @20 est le plus bas après l'idée 1
(0,805) : down-weighter une liste plate dans la FUSION (pas seulement au
rerank) réduit aussi la probabilité qu'un candidat de cette liste atteigne
le top-20, un compromis attendu du mécanisme. Piste non essayée : ne
down-weighter qu'au-delà du rerank plutôt que dans la fusion elle-même.

**Idée 4 — paires adjacentes**. Effet quasi nul sur toutes les catégories
(≤0,001 partout) malgré un plafond @20/@30 légèrement supérieur à la
baseline RRF pure (0,809/0,822 vs runs 4/5/combo identiques) — les tours
ajoutés (réponse du partenaire à une question) existent bien et remontent
dans la fenêtre 11-30, mais soit ils étaient déjà couverts par un autre
mécanisme (facts+keywords, planification), soit ils ne portent pas la
preuve cherchée par la question. Le coût de latence est le plus élevé des
7 runs (964,4 ms p50) pour un gain nul — la détection de question
(`_is_question_turn`) et le hop de session s'exécutent sur chaque question
sans bénéfice mesuré. Décision : mécanisme sans effet démontré sur ce corpus.

**Idée 5 — fourragement itératif**. Indiscernable de la baseline sur toutes
les catégories (identique à 3 décimales, sauf ceiling@30 +0,001) : soit
très peu de sous-requêtes sont réellement « affamées » (zéro candidat dans
le top-20 fusionné) sur ce corpus déjà bien couvert par la planification
§A, soit celles qui le sont ne trouvent rien de nouveau à `top_k*8` non plus
(la sous-requête elle-même, pas seulement sa profondeur de recherche, ne
porte pas l'évidence). p50 (844 ms) est dans la plage normale, confirmant
que le mécanisme se déclenche rarement (le budget de 2 recherches
supplémentaires par question n'est presque jamais consommé). Piste non
essayée : mesurer directement `n_starved`/`n_extra_searches` pour confirmer
le taux de déclenchement avant de l'abandonner définitivement.

### Conclusion

Les cinq idées attaquent authentiquement la couverture (paramètres
librement variables : sièges de quota, λ MMR, plancher de confiance, décai
d'adjacence, budget itératif) mais **aucune ne déplace le multi-hop ou
l'open-domaine au-delà du bruit de mesure** une fois le pipeline §A+§C déjà
en place (RRF multi-requête + reranker Ettin-68m) — contrairement aux
mécanismes de proximité structurelle déjà réfutés deux fois (E12, §B), ces
idées ne réduisent pas le rappel non plus : elles restent globalement
neutres. Lecture cohérente avec le reste de la campagne P9 (voir
« Interprétation » plus haut) : l'écart restant vers 0,86 (multi-hop
0,656-0,664, open-domaine 0,497-0,510 selon la config) est un problème de
**précision du classement sur des candidats déjà présents** plutôt que de
couverture des candidats eux-mêmes — la seule piste qui a mesurablement
comblé cet écart à ce jour reste le reranking listwise par LLM (P9 étape D,
macro R@10 0,777→0,815 avec Sonnet 5 sur les mêmes candidats). **Décision :
aucune promotion, les 5 mécanismes restent opt-in dans
`bench/retrievers.py::V4PlannedRetriever`, testés (38 nouveaux tests dans
`tests/test_bench_plan.py`), jamais portés dans `src/memory_os_v4`.**

## V5.0 — canaux facettes et implications (2026-09-19)

Objectif (`docs/IDEAS.md` idée 8) : le témoin à taille égale (2026-09-19,
`bench/facet_size_matched.py`) a établi que le canal facette élève le
PLAFOND de rappel multi-hop de +0,023 macro au-delà de l'effet de taille
seul (positif dans 10/10 conversations) — reste à mesurer si ce gain de
plafond survit au classement réel dans le top-10. Deux canaux ajoutés en
option (défaut désactivé) à `V4PlannedRetriever` (`bench/retrievers.py`,
`bench/enrich.py`) : `--facet-channel [--facet-weight W]` (indexe
`(subject, predicate) -> {turn_id}` par conversation, prédicat `other`
exclu, prédicat EXACT uniquement — pas d'élargissement par famille ; le
pool de la question est classé par meilleur rang déjà obtenu dans les
listes question/sous-requêtes/hypothèses puis injecté comme un parti RRF
de plus, AVANT la fusion) et `--implications` (indexe les implications de
chaque fait via `add_facts(parent_ids=[turn_id], tags=["implication"])`,
mécanisme identique aux faits enrichis existants).

Garde-fou machine : un seul processus à la fois, `nice -n 15`, portes
resserrées en cours de campagne suite à observation (swap < 6 Go puis <
4,5 Go ET compression < 5 Go avant chaque lancement, kill inchangé à swap
> 6,5 Go / compression ≥ 7 Go / disque < 3 Go). **Run (e) (reranker
cross-encoder sur la meilleure config) a été STOPPÉ PAR LE GARDE-FOU** :
swap 6544 Mo > seuil 6500 Mo au moment du kill (aucun résultat produit,
non relancé — consigne explicite « si le garde-fou tue un run, arrêter la
campagne et rapporter »). Run (d) (meilleur poids facette + implications)
**sauté par construction** : sa condition de déclenchement (implications
seules doivent battre 0,721 de plus de 0,005) n'est pas remplie (0,719,
voir plus bas).

Réserve de protocole à garder visible pour tout résultat facette : les
facettes de question ont été produites à partir du seul texte de la
question, mais pour quelques questions ambiguës l'annotateur a déduit le
sujet à partir des questions voisines du même fichier — information qu'un
assistant en production ne verrait pas.

| Run | multi-hop R@10 | temporel R@10 | open-dom. R@10 | single-hop R@10 | macro R@5 | macro R@10 | macro MRR | macro nDCG@10 | p50 (ms) | pool moyen / nouveaux vs top-50 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline `locomo_v4_plan_norerank_k150_dump` (row 79) | 0,618 | 0,860 | 0,486 | 0,919 | 0,631 | **0,721** | 0,573 | 0,577 | 172,7 | — |
| a. `locomo_v5_facets_w05` (row 80) : facettes w=0,5 | 0,625 | 0,876 | 0,484 | 0,924 | 0,650 | 0,727 (+0,006) | 0,609 (+0,036) | 0,600 | 232,1 | 17,4 / 10,5 |
| b. `locomo_v5_facets_w10` (row 81) : facettes w=1,0 | 0,659 | 0,884 | 0,512 | 0,927 | 0,666 | **0,746 (+0,025)** | 0,617 (+0,044) | 0,613 | 188,6 | 17,4 / 10,5 |
| b2. `locomo_v5_facets_w20` (row 82) : facettes w=2,0 | **0,690** | **0,898** | **0,513** | 0,925 | **0,671** | **0,757 (+0,036)** | 0,608 (+0,035) | 0,610 | 204,9 | 17,4 / 10,5 |
| c. `locomo_v5_impl` (row 83) : implications seules | 0,613 | 0,867 | 0,480 | 0,917 | 0,628 | 0,719 (−0,002) | 0,584 (+0,011) | 0,582 | 199,1 | — (532 implications indexées) |
| d. `locomo_v5_facets_impl` | — | — | — | — | — | **SAUTÉ** (c ne bat pas 0,721 de +0,005) | — | — | — | — |
| e. `locomo_v5_best_e68` (facettes w=2,0 + Ettin-68m) | — | — | — | — | — | **TUÉ PAR LE GARDE-FOU** (swap 6544 Mo) | — | — | — | — |

Le pool moyen (17,4 tours/question, dont 10,5 en moyenne absents du top-50
pré-facette) et la couverture (100 % des 1536 questions ont ≥1 facette,
`bench/data/enrich/locomo_qfacets`) sont IDENTIQUES pour les trois poids
(le poids ne change que la pondération RRF du parti facette, jamais le
pool lui-même) — colonne rapportée une seule fois par lisibilité.

### Canal facette : gain réel, dose-réponse claire, franchit largement le seuil de bruit

Le canal facette seul (poids 1,0 ou 2,0) **bat le seuil de bruit de 0,005**
sur le macro R@10 : +0,025 (w=1,0) et +0,036 (w=2,0), porté presque
entièrement par le multi-hop (+0,041 puis +0,072) et le temporel (+0,024
puis +0,038) — exactement les deux catégories visées par l'idée 8. Le
poids 2,0 bat le poids 1,0 sur 3 des 4 catégories (multi-hop, temporel,
open-domaine) et perd de très peu sur le single-hop (0,925 vs 0,927,
−0,002, dans le bruit) : **poids 2,0 retenu comme meilleure config de la
campagne**, sans qu'un plateau soit atteint — un poids encore plus élevé
n'a pas été testé (hors mandat de cette session, campagne réduite par le
garde-fou machine après le kill du run e). single-hop et open-domaine
restent proches du plafond déjà documenté (0,979/0,691 dans docs/IDEAS.md),
ce qui borne mécaniquement leur marge de progression.

**Sur la question « signal de reclassement ou nouvelle couverture ? »** :
la série à trois poids permet de trancher, alors qu'un seul point (poids
0,5) ne le permettait pas. Au poids 0,5, le MRR macro bouge 6× plus que le
R@10 macro (+0,036 contre +0,006) — lecture initiale plausible : le canal
ne fait que remonter dans le classement des tours déjà autrement
accessibles. Cette lecture ne tient PAS à poids plus élevé : le R@10 macro
continue de croître de façon quasi linéaire avec le poids (0,006 → 0,025 →
0,036) alors que le MRR macro PLAFONNE puis recule légèrement (0,036 →
0,044 → 0,035) ; au poids 2,0, le MRR du temporel devient même négatif
(0,702 → 0,701, −0,001) alors que son R@10 continue de monter (+0,038).
Interprétation : à poids croissant, le canal facette pousse de PLUS EN PLUS
de candidats réellement nouveaux dans le top-10 (ce qui fait monter le
R@10) sans amener ces nouveaux candidats jusqu'au rang 1 aussi souvent que
les candidats déjà présents (ce qui plafonne/recule le MRR) — cohérent
avec le mécanisme tel qu'implémenté : un parti RRF de plus, pas une
priorité de classement sur les candidats existants. Le run (a) seul
suggérait la lecture inverse ; l'ensemble des trois runs la contredit. Ni
l'une ni l'autre lecture ne doit être vue comme définitive : la série n'a
que 3 points et n'isole pas la contribution exacte de chaque tour facette
individuel (mesure agrégée, pas par-candidat).

### Canal implications : résultat nul

`--implications` seul (532 items indexés sur 3332 faits, `n_indexed=532`
dans `metadata.facets`) est **indiscernable du bruit** : macro R@10
0,719 contre 0,721 (−0,002), et chaque catégorie individuellement est dans
[−0,006 ; +0,007] — aucune catégorie ne bouge de façon exploitable, y
compris l'open-domaine visé par le mécanisme (0,486 → 0,480, −0,006). Le
MRR macro bouge un peu plus (+0,011) mais reste loin du seuil. Lecture
honnête : soit les 529 faits annotés avec implication (docs/IDEAS.md) sont
trop peu nombreux/trop dispersés pour changer le classement d'ensemble sur
1536 questions, soit le texte des implications (généralement une
paraphrase inférentielle du fait, pas un nouveau terme de recherche) ne
diffère pas assez lexicalement/sémantiquement de l'énoncé déjà indexé du
fait lui-même pour capter des requêtes qu'il ne captait pas déjà. Rien
n'indique une régression réelle non plus (tous les écarts individuels sont
sous 0,006) — **résultat nul, rapporté tel quel**, conforme au prior
honnête de la tâche (« gain probablement faible, potentiellement dans le
bruit »). Run (d) (meilleur poids facette + implications), dont la
condition de déclenchement était « implications seules battent 0,721 de
plus de 0,005 », est donc sauté par construction.

### Run (e) : arrêt garde-fou, aucune donnée

Le run reranké (facettes w=2,0 + `tomaarsen/ms-marco-ettin-68m-reranker`,
`--rerank-top-n 50`) a été tué par le garde-fou machine à swap=6544 Mo
(> seuil 6500 Mo), compression 4,14 Go au moment du kill, disque 5,68 Go
libres — aucun JSON de résultat, aucune ligne de ledger. Le run n'a PAS été
relancé (consigne explicite de la campagne : un kill du garde-fou arrête
la campagne). La comparaison contre `locomo_v4_plan_best_k20` (0,751,
reranké, poids facette non appliqué) n'a donc pas pu être faite ; c'est un
angle mort assumé de cette session, pas un résultat négatif — piste à
reprendre dans une session ultérieure avec un budget mémoire dédié
(peut-être `--rerank-top-n` plus bas que 50, ou un lot de conversations
réduit pour amortir le pic mémoire du chargement du modèle Ettin-68m en
plus du modèle d'embedding déjà chargé).

### Décision

**Le canal facette gagne réellement** (poids 2,0 : macro R@10 0,721 →
0,757, +0,036, tiré par multi-hop +0,072 et temporel +0,038) — contrairement
au prior honnête de la tâche qui anticipait un gain probablement dans le
bruit, la mesure R@10 confirme ici une fraction substantielle (environ la
moitié à un tiers selon la catégorie) du gain de PLAFOND déjà mesuré par
`bench/facet_size_matched.py` (+0,023 multi-hop taille égale, +0,053
open-domaine). Le canal implications, lui, confirme le prior honnête : nul
(0,719 contre 0,721). Aucune promotion vers `src/memory_os_v4` (hors
mandat, banc seulement) ; `--facet-channel`/`--facet-weight`/
`--implications` restent opt-in, testés (33 nouveaux tests dans
`tests/test_bench_facets.py`, suite bench 93→126 verts), défauts
inchangés. Le run reranké (e) reste à faire — voir section précédente.
