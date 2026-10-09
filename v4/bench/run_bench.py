"""CLI: run one retriever against one dataset, score it, and record it.

    cd v4 && PYTHONPATH=src:. ../.venv/bin/python -m bench.run_bench \\
        --dataset locomo --retriever v3equiv --name locomo_v3equiv --top-k 10

For each BenchConversation: reset the retriever, index its items once, then
run every question's query, timing each search() call. Writes
bench/results/<name>_v<N>.json (N auto-increments, nothing is ever
overwritten) and appends exactly one row to the RESULTS.md ledger (unless
--no-ledger, used by tests so they never touch the real ledger).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from bench.enrich import MAX_DUMP_CANDIDATES, build_candidate_dump_line, load_candidate_dump, load_llm_rankings
from bench.locomo import DEFAULT_PATH as LOCOMO_DEFAULT_PATH
from bench.locomo import load_locomo
from bench.longmemeval import DEFAULT_PATH as LME_DEFAULT_PATH
from bench.longmemeval import iter_longmemeval
from bench.metrics import aggregate, context_tokens, mrr, ndcg_at_k, recall_at_k
from bench.protocol import BenchConversation
from bench.retrievers import (
    EMBED_MODEL_NAME,
    BM25TurnRetriever,
    DenseTurnRetriever,
    V3EquivRetriever,
    V3ExactRetriever,
    V4EnrichedRetriever,
    V4LLMRerankedRetriever,
    V4PlannedRetriever,
    V4Retriever,
    apply_llm_ranking,
)

BENCH_DIR = Path(__file__).resolve().parent
V4_DIR = BENCH_DIR.parent
DEFAULT_RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_LEDGER_PATH = V4_DIR / "RESULTS.md"
DEFAULT_ENRICH_DIR = BENCH_DIR / "data" / "enrich" / "locomo_facts"
DEFAULT_QUERIES_DIR = BENCH_DIR / "data" / "enrich" / "locomo_queries"
# Idea 8 (docs/IDEAS.md, 2026-09-19): facet channel + implications channel.
DEFAULT_FACETS_DIR = BENCH_DIR / "data" / "enrich" / "locomo_facets"
DEFAULT_QFACETS_DIR = BENCH_DIR / "data" / "enrich" / "locomo_qfacets"


def _iso_date(event_time: Optional[float]) -> Optional[str]:
    """`item.event_time` (a UTC unix timestamp, see `bench.locomo`) ->
    ``"YYYY-MM-DD"``, or None if there is no timestamp (LongMemEval turns,
    or a LoCoMo session date that failed to parse). Used only by
    `--dump-candidates` (P9 step D)."""
    if event_time is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(event_time, tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Retriever construction
# ---------------------------------------------------------------------------
def make_retriever(
    retriever_key: str,
    config_overrides: dict,
    name: str,
    enrich_dir: Optional[str] = None,
    enrich_mode: str = "facts",
    plan_kwargs: Optional[dict] = None,
    llmrank_dir: Optional[str] = None,
    candidates_dir: Optional[str] = None,
):
    if retriever_key == "v3equiv":
        return V3EquivRetriever(name=name)
    if retriever_key == "v3exact":
        return V3ExactRetriever(name=name)
    if retriever_key == "dense":
        return DenseTurnRetriever(name=name)
    if retriever_key == "bm25":
        return BM25TurnRetriever(name=name)
    if retriever_key == "v4":
        return V4Retriever(config_overrides=config_overrides, name=name)
    if retriever_key == "v4_enriched":
        return V4EnrichedRetriever(
            config_overrides=config_overrides,
            name=name,
            mode=enrich_mode,
            enrich_dir=enrich_dir or str(DEFAULT_ENRICH_DIR),
        )
    if retriever_key == "v4_planned":
        return V4PlannedRetriever(
            config_overrides=config_overrides,
            name=name,
            mode=enrich_mode,
            enrich_dir=enrich_dir or str(DEFAULT_ENRICH_DIR),
            **(plan_kwargs or {}),
        )
    if retriever_key == "v4_llmrank":
        return V4LLMRerankedRetriever(
            config_overrides=config_overrides,
            name=name,
            mode=enrich_mode,
            enrich_dir=enrich_dir or str(DEFAULT_ENRICH_DIR),
            llmrank_dir=llmrank_dir,
            candidates_dir=candidates_dir,
            **(plan_kwargs or {}),
        )
    raise ValueError(f"unknown --retriever {retriever_key!r}")


def stratified_sample(convs: list[BenchConversation], limit: int) -> list[BenchConversation]:
    """Round-robin `convs` across each conversation's question_type/category
    (`conv.questions[0].category`) until `limit` is reached, so a small
    --limit still spans every category instead of just the first N
    conversations in file order. Deterministic: categories are visited in
    first-seen order, and each category's own conversations keep their
    original relative order. One conversation per category per round (this
    is only meaningful for datasets with exactly one question per
    BenchConversation, e.g. LongMemEval oracle — see E7, docs/ABLATIONS.md).
    """
    order: list[str] = []
    queues: dict[str, list[BenchConversation]] = {}
    for conv in convs:
        cat = conv.questions[0].category if conv.questions else "unknown"
        if cat not in queues:
            queues[cat] = []
            order.append(cat)
        queues[cat].append(conv)

    out: list[BenchConversation] = []
    while len(out) < limit and any(queues[c] for c in order):
        for cat in order:
            if len(out) >= limit:
                break
            if queues[cat]:
                out.append(queues[cat].pop(0))
    return out[:limit]


def load_dataset(
    dataset: str,
    limit: Optional[int],
    locomo_path: Optional[str] = None,
    lme_path: Optional[str] = None,
    stratified: bool = False,
) -> Iterable[BenchConversation]:
    """Yield BenchConversations. --limit means first N conversations (LoCoMo)
    or first N questions (LongMemEval oracle: 1 question == 1 conversation).

    `stratified` (E7, docs/ABLATIONS.md): round-robin the selection across
    question_type/category instead of taking a plain file-order prefix. Only
    meaningful for longmemeval_oracle (one question per BenchConversation);
    for locomo each conversation already carries hundreds of mixed-category
    questions, so `stratified` is a no-op there (printed once via caller).
    """
    if dataset == "locomo":
        convs = load_locomo(locomo_path or LOCOMO_DEFAULT_PATH)
        return convs[:limit] if limit else convs
    if dataset == "longmemeval_oracle":
        if stratified:
            all_convs = list(iter_longmemeval(lme_path or LME_DEFAULT_PATH, limit=None))
            return stratified_sample(all_convs, limit) if limit else all_convs
        return iter_longmemeval(lme_path or LME_DEFAULT_PATH, limit=limit)
    raise ValueError(f"unknown --dataset {dataset!r}")


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------
def run_bench(
    dataset: str,
    retriever_key: str,
    name: str,
    top_k: int = 10,
    limit: Optional[int] = None,
    config_overrides: Optional[dict] = None,
    seed: int = 0,
    locomo_path: Optional[str] = None,
    lme_path: Optional[str] = None,
    stratified: bool = False,
    enrich_dir: Optional[str] = None,
    enrich_mode: str = "facts",
    allow_partial_enrich: bool = False,
    queries_dir: Optional[str] = None,
    plan_mode: str = "sub+hyp",
    weight_original: float = 1.0,
    weight_sub: float = 1.0,
    weight_hyp: float = 0.7,
    rrf_k: int = 60,
    rerank_model: Optional[str] = None,
    rerank_device: str = "auto",
    rerank_top_n: int = 30,
    plan_rerank_blend: bool = True,
    fact_links: bool = False,
    link_seed: int = 8,
    link_decay: float = 0.5,
    allow_partial_plan: bool = False,
    llmrank_dir: Optional[str] = None,
    candidates_dir: Optional[str] = None,
    dump_candidates_dir: Optional[str] = None,
    fusion_mode: str = "rrf",
    quota_then_rrf_boundary: int = 10,
    subquery_mmr: bool = False,
    mmr_lambda: float = 0.6,
    signal_confidence: bool = False,
    adjacency_pair: bool = False,
    iterative_budget: bool = False,
    report_ceilings: bool = False,
    facet_channel: bool = False,
    facet_weight: float = 0.5,
    facets_dir: Optional[str] = None,
    qfacets_dir: Optional[str] = None,
    implications: bool = False,
) -> dict[str, Any]:
    import random

    random.seed(seed)
    config_overrides = dict(config_overrides or {})
    # v4_llmrank (P9 step D, PLAN.md) IS-A V4PlannedRetriever (bench.retrievers)
    # and needs the exact same plan/enrich/rerank construction kwargs.
    is_planned = retriever_key in ("v4_planned", "v4_llmrank")
    is_llmrank = retriever_key == "v4_llmrank"
    plan_kwargs = (
        {
            "queries_dir": queries_dir or str(DEFAULT_QUERIES_DIR),
            "plan_mode": plan_mode,
            "weight_original": weight_original,
            "weight_sub": weight_sub,
            "weight_hyp": weight_hyp,
            "rrf_k": rrf_k,
            "rerank_model": rerank_model,
            "rerank_device": rerank_device,
            "rerank_top_n": rerank_top_n,
            "plan_rerank_blend": plan_rerank_blend,
            "fact_links": fact_links,
            "link_seed": link_seed,
            "link_decay": link_decay,
        }
        if is_planned
        else None
    )
    if retriever_key == "v4_planned":
        # Idées inter-disciplines (docs/IDEAS.md, 2026-09-14): V4PlannedRetriever
        # only -- V4LLMRerankedRetriever's __init__ does not forward these
        # (out of scope for P9 step D, left untouched per the task brief).
        plan_kwargs.update(
            {
                "fusion_mode": fusion_mode,
                "quota_then_rrf_boundary": quota_then_rrf_boundary,
                "subquery_mmr": subquery_mmr,
                "mmr_lambda": mmr_lambda,
                "signal_confidence": signal_confidence,
                "adjacency_pair": adjacency_pair,
                "iterative_budget": iterative_budget,
                # Idea 8 (docs/IDEAS.md, 2026-09-19): V4PlannedRetriever only,
                # same "out of scope for v4_llmrank" rule as the block above.
                "facet_channel": facet_channel,
                "facet_weight": facet_weight,
                "facets_dir": facets_dir or str(DEFAULT_FACETS_DIR),
                "qfacets_dir": qfacets_dir or str(DEFAULT_QFACETS_DIR),
                "implications": implications,
            }
        )
    retriever = make_retriever(
        retriever_key,
        config_overrides,
        name,
        enrich_dir=enrich_dir,
        enrich_mode=enrich_mode,
        plan_kwargs=plan_kwargs,
        llmrank_dir=llmrank_dir,
        candidates_dir=candidates_dir,
    )

    # V4PlannedRetriever/V4LLMRerankedRetriever IS-A V4EnrichedRetriever
    # (bench.retrievers): all three carry `.coverage`/`.search_stats` for the
    # offline-fact-enrichment bookkeeping below; v4_planned/v4_llmrank
    # additionally carry `.plan_stats`, and v4_llmrank alone also carries
    # `.llm_stats` (P9 step D).
    is_enriched = retriever_key in ("v4_enriched", "v4_planned", "v4_llmrank")
    enrich_dir_path = Path(enrich_dir) if enrich_dir else DEFAULT_ENRICH_DIR
    queries_dir_path = Path(queries_dir) if queries_dir else DEFAULT_QUERIES_DIR
    dump_candidates_dir_path = Path(dump_candidates_dir) if dump_candidates_dir else None

    convs = list(
        load_dataset(dataset, limit, locomo_path=locomo_path, lme_path=lme_path, stratified=stratified)
    )

    if is_enriched:
        # Fail fast (before any indexing/searching happens) if the enrich
        # dir is missing files for conversations this run would need, unless
        # the caller explicitly opted into a partial run.
        missing = [c.conv_id for c in convs if not (enrich_dir_path / f"{c.conv_id}.jsonl").exists()]
        if missing and not allow_partial_enrich:
            raise RuntimeError(
                f"--enrich-dir {enrich_dir_path} has no enrichment file for "
                f"{len(missing)}/{len(convs)} conversation(s): {missing}. "
                "Pass --allow-partial-enrich to run anyway (coverage will be recorded "
                "in the results JSON metadata and the RESULTS.md Notes column)."
            )

    if is_planned:
        # Same fail-fast pattern as --allow-partial-enrich above, but for the
        # PER-QUESTION query-plan files (bench/data/enrich/locomo_queries/):
        # a missing file here still lets every question in that conversation
        # run (V4PlannedRetriever falls back to single-query search per
        # missing qid, see plan_stats["n_fallback_missing_plan"]), so this is
        # a conversation-level early-warning, not a strict requirement.
        missing_plan = [c.conv_id for c in convs if not (queries_dir_path / f"{c.conv_id}.jsonl").exists()]
        if missing_plan and not allow_partial_plan:
            raise RuntimeError(
                f"--queries-dir {queries_dir_path} has no query-plan file for "
                f"{len(missing_plan)}/{len(convs)} conversation(s): {missing_plan}. "
                "Pass --allow-partial-plan to run anyway (every question in a missing "
                "conversation falls back to single-query search; coverage is recorded "
                "in the results JSON metadata and the RESULTS.md Notes column)."
            )

    rows: list[dict[str, Any]] = []
    n_conversations = 0
    n_items_indexed = 0
    index_build_time_s = 0.0
    enrichment_totals = {
        "n_turns": 0,
        "n_turns_with_facts": 0,
        "n_facts_added": 0,
        "n_conversations_with_file": 0,
    }
    dump_totals = {"n_files": 0, "n_lines": 0}
    for conv in convs:
        n_conversations += 1
        retriever.reset()
        if conv.items:
            # Palier 2-3 (E10, docs/ABLATIONS.md): time every add() call --
            # dominated by embedding encode cost -- to report encode
            # throughput (turns/s) and total index build time per run,
            # regardless of --dataset/--retriever.
            t_add0 = time.perf_counter()
            retriever.add(conv.items)
            index_build_time_s += time.perf_counter() - t_add0
            n_items_indexed += len(conv.items)
            if is_enriched:
                cov = retriever.coverage
                enrichment_totals["n_turns"] += cov["n_turns"]
                enrichment_totals["n_turns_with_facts"] += cov["n_turns_with_facts"]
                enrichment_totals["n_facts_added"] += cov["n_facts_added"]
                enrichment_totals["n_conversations_with_file"] += int(cov["enrich_file_found"])
        text_by_id = {item.id: item.text for item in conv.items}
        # --dump-candidates only (P9 step D): built once per conversation,
        # not per question, since they're the same for every question in it.
        date_by_id = {item.id: _iso_date(item.event_time) for item in conv.items} if dump_candidates_dir_path else {}
        speaker_by_id = {item.id: item.speaker for item in conv.items} if dump_candidates_dir_path else {}
        facts_for_turn = getattr(retriever, "facts_for_turn", None)
        dump_lines: list[str] = []

        for q in conv.questions:
            search_kwargs: dict[str, Any] = {"top_k": top_k, "at_time": q.query_time}
            if getattr(retriever, "accepts_qid", False):
                # V4PlannedRetriever (bench.retrievers): the base 3-arg
                # Retriever protocol has no `qid`, but this retriever needs
                # it to look up the question's precomputed query plan.
                search_kwargs["qid"] = q.qid
            if dump_candidates_dir_path is not None:
                # Oversample to MAX_DUMP_CANDIDATES (20) so one search() call
                # serves both: `ranked[:top_k]` below is IDENTICAL to what a
                # plain `--top-k top_k` run would have scored (metrics stay
                # comparable), while `ranked[:MAX_DUMP_CANDIDATES]` is the
                # FINAL top-20 the dump needs -- no second search() call, no
                # doubled latency.
                search_kwargs["top_k"] = max(top_k, MAX_DUMP_CANDIDATES)
            if report_ceilings:
                # Idées inter-disciplines (docs/IDEAS.md): one oversampled
                # search() call also serves the recall CEILINGS (evidence is
                # used only to SCORE this already-computed ranked list
                # afterwards -- never fed back into retrieval) at k in
                # (10, 20, 30, 50), same "max(...) so metrics stay
                # comparable" pattern as --dump-candidates above.
                search_kwargs["top_k"] = max(search_kwargs["top_k"], 50)
            t0 = time.perf_counter()
            ranked_full = retriever.search(q.text, **search_kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            # was hardcoded to ranked[:10] regardless of --top-k (adversarial review, 2026-09-12)
            ranked = ranked_full[:top_k]
            top_texts = [text_by_id.get(i, "") for i in ranked]

            if dump_candidates_dir_path is not None:
                dump_ranked = ranked_full[:MAX_DUMP_CANDIDATES]
                facts_by_id = {tid: (facts_for_turn(tid) if facts_for_turn else []) for tid in dump_ranked}
                dump_lines.append(
                    json.dumps(
                        build_candidate_dump_line(
                            qid=q.qid,
                            question=q.text,
                            ranked_turn_ids=dump_ranked,
                            text_by_id=text_by_id,
                            date_by_id=date_by_id,
                            speaker_by_id=speaker_by_id,
                            facts_by_id=facts_by_id,
                        ),
                        ensure_ascii=False,
                    )
                )

            row = {
                "qid": q.qid,
                "category": q.category,
                "r5": recall_at_k(ranked, q.evidence_ids, 5),
                "r10": recall_at_k(ranked, q.evidence_ids, 10),
                "mrr": mrr(ranked, q.evidence_ids),
                "ndcg10": ndcg_at_k(ranked, q.evidence_ids, 10),
                "latency_ms": latency_ms,
                "context_tokens": context_tokens(top_texts),
            }
            if report_ceilings:
                for k in (10, 20, 30, 50):
                    row[f"ceiling_r{k}"] = recall_at_k(ranked_full[:50], q.evidence_ids, k)
            rows.append(row)

        if dump_candidates_dir_path is not None and dump_lines:
            dump_candidates_dir_path.mkdir(parents=True, exist_ok=True)
            (dump_candidates_dir_path / f"{conv.conv_id}.jsonl").write_text(
                "\n".join(dump_lines) + "\n", encoding="utf-8"
            )
            dump_totals["n_files"] += 1
            dump_totals["n_lines"] += len(dump_lines)

    agg = aggregate(rows)
    ceilings_agg = (
        aggregate(rows, metric_keys=("ceiling_r10", "ceiling_r20", "ceiling_r30", "ceiling_r50"))
        if report_ceilings
        else None
    )

    import platform
    from datetime import datetime, timezone

    result = {
        "name": name,
        "dataset": dataset,
        "retriever": retriever_key,
        "config": {
            "top_k": top_k,
            "limit": limit,
            "seed": seed,
            "stratified": stratified,
            "config_overrides": config_overrides,
        },
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "embed_model": EMBED_MODEL_NAME if retriever_key in ("v3equiv", "v3exact", "dense") else config_overrides.get("embed_model", EMBED_MODEL_NAME),
            # recorded per adversarial review (2026-09-12): candidate-set
            # iteration order in bm25.py/entities.py/retrieval.py depends on
            # PYTHONHASHSEED; both are now made fully deterministic via
            # explicit (score, id) tiebreaks regardless of this value, but it
            # is still recorded for transparency/reproducibility.
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            # Palier 2-3 (E10): total add()/index-build wall time across all
            # conversations, and the derived encode throughput (turns/s) --
            # dominated by embedding encode cost for retrievers that embed.
            "index_build_time_s": index_build_time_s,
            "n_items_indexed": n_items_indexed,
            "encode_throughput_items_per_s": (
                n_items_indexed / index_build_time_s if index_build_time_s > 0 else None
            ),
            # V4EnrichedRetriever only: coverage of the offline LLM
            # enrichment actually consumed by this run, plus how often
            # search()'s top raw hit came from a fact vs. an episode item
            # (bench.retrievers.V4EnrichedRetriever.search_stats).
            "enrichment": (
                {
                    "enrich_dir": str(enrich_dir_path),
                    "enrich_mode": enrich_mode,
                    "allow_partial_enrich": allow_partial_enrich,
                    "n_conversations": n_conversations,
                    "n_conversations_with_file": enrichment_totals["n_conversations_with_file"],
                    "n_conversations_missing_file": (
                        n_conversations - enrichment_totals["n_conversations_with_file"]
                    ),
                    "turns_total": enrichment_totals["n_turns"],
                    "turns_with_facts": enrichment_totals["n_turns_with_facts"],
                    "coverage_turns": (
                        enrichment_totals["n_turns_with_facts"] / enrichment_totals["n_turns"]
                        if enrichment_totals["n_turns"]
                        else 0.0
                    ),
                    "facts_added": enrichment_totals["n_facts_added"],
                }
                if is_enriched
                else None
            ),
            "search_stats": dict(retriever.search_stats) if is_enriched else None,
            # V4PlannedRetriever only (P9 §A): coverage of the offline query
            # plans actually consumed by this run (bench.retrievers.
            # V4PlannedRetriever.plan_stats) plus the resolved plan config.
            "planning": (
                {
                    "queries_dir": str(queries_dir_path),
                    "plan_mode": plan_mode,
                    "weight_original": weight_original,
                    "weight_sub": weight_sub,
                    "weight_hyp": weight_hyp,
                    "rrf_k": rrf_k,
                    "rerank_model": rerank_model,
                    "rerank_top_n": rerank_top_n,
                    "plan_rerank_blend": plan_rerank_blend,
                    "fact_links": fact_links,
                    "link_seed": link_seed,
                    "link_decay": link_decay,
                    "allow_partial_plan": allow_partial_plan,
                    # Idées inter-disciplines (docs/IDEAS.md, 2026-09-14):
                    # recorded even when off (all default False/"rrf") so a
                    # results JSON is self-describing without cross-checking
                    # the CLI invocation that produced it.
                    "fusion_mode": fusion_mode,
                    "quota_then_rrf_boundary": quota_then_rrf_boundary,
                    "subquery_mmr": subquery_mmr,
                    "mmr_lambda": mmr_lambda,
                    "signal_confidence": signal_confidence,
                    "adjacency_pair": adjacency_pair,
                    "iterative_budget": iterative_budget,
                    **dict(retriever.plan_stats),
                    "fraction_questions_with_plan": (
                        retriever.plan_stats["n_plan_used"] / retriever.plan_stats["n_searches"]
                        if retriever.plan_stats["n_searches"]
                        else 0.0
                    ),
                }
                if is_planned
                else None
            ),
            # --retriever v4_planned only (idea 8, docs/IDEAS.md, 2026-09-19):
            # facet-channel coverage + implications-channel indexing totals
            # (bench.retrievers.V4PlannedRetriever.facet_stats), recorded
            # even when both options are off (facet_stats then reads all
            # zeros) so a results JSON is self-describing.
            "facets": (
                {
                    "facet_channel": facet_channel,
                    "facet_weight": facet_weight,
                    "facets_dir": str(Path(facets_dir) if facets_dir else DEFAULT_FACETS_DIR),
                    "qfacets_dir": str(Path(qfacets_dir) if qfacets_dir else DEFAULT_QFACETS_DIR),
                    "implications": implications,
                    "n_questions": retriever.facet_stats["n_searches"],
                    "n_questions_with_facets": retriever.facet_stats["n_questions_with_facets"],
                    "fraction_questions_with_facets": (
                        retriever.facet_stats["n_questions_with_facets"] / retriever.facet_stats["n_searches"]
                        if retriever.facet_stats["n_searches"]
                        else 0.0
                    ),
                    # Mean over EVERY question the facet channel ran for
                    # (including ones with zero facets, which contribute a
                    # pool size of 0) -- mirrors bench/facet_size_matched.py's
                    # own "new candidates per question" convention.
                    "mean_pool_size": (
                        retriever.facet_stats["total_pool_size"] / retriever.facet_stats["n_searches"]
                        if retriever.facet_stats["n_searches"]
                        else 0.0
                    ),
                    "mean_new_vs_fused_top50": (
                        retriever.facet_stats["total_new_vs_top50"] / retriever.facet_stats["n_searches"]
                        if retriever.facet_stats["n_searches"]
                        else 0.0
                    ),
                    "n_implications_indexed": retriever.facet_stats["n_implications_indexed"],
                }
                if retriever_key == "v4_planned"
                else None
            ),
            # V4LLMRerankedRetriever only (P9 step D): coverage of the
            # offline LLM listwise reranking actually consumed by this run
            # (bench.retrievers.V4LLMRerankedRetriever.llm_stats).
            "llm_rerank": (
                {
                    "llmrank_dir": str(Path(llmrank_dir)) if llmrank_dir else None,
                    "candidates_dir": str(Path(candidates_dir)) if candidates_dir else None,
                    **dict(retriever.llm_stats),
                    "fraction_llm_ranked": (
                        retriever.llm_stats["llm_ranked_questions"] / retriever.llm_stats["n_searches"]
                        if retriever.llm_stats["n_searches"]
                        else 0.0
                    ),
                }
                if is_llmrank
                else None
            ),
            # --dump-candidates only (P9 step D): where the FINAL top-20
            # per-question candidate dumps were written, and how many.
            "dump_candidates": (
                {
                    "dir": str(dump_candidates_dir_path),
                    "n_files": dump_totals["n_files"],
                    "n_lines": dump_totals["n_lines"],
                    "max_candidates": MAX_DUMP_CANDIDATES,
                }
                if dump_candidates_dir_path is not None
                else None
            ),
            # --report-ceilings only (idées inter-disciplines, docs/IDEAS.md):
            # recall@k (k in 10/20/30/50) of the FINAL ranked list from a
            # single `search(top_k=50)` call per question -- evidence is
            # used only to SCORE this list here, never fed back into
            # retrieval. per_category/macro/micro mirror the main metrics'
            # own shape (bench.metrics.aggregate), keyed "r10"/"r20"/"r30"/"r50"
            # (the "ceiling_" prefix is stripped for readability).
            "ceilings": (
                {
                    "per_category": {
                        cat: {f"r{k}": vals[f"ceiling_r{k}"] for k in (10, 20, 30, 50)} | {"n": vals["n"]}
                        for cat, vals in ceilings_agg["per_category"].items()
                    },
                    "macro": {f"r{k}": ceilings_agg["macro"][f"ceiling_r{k}"] for k in (10, 20, 30, 50)},
                    "micro": {f"r{k}": ceilings_agg["micro"][f"ceiling_r{k}"] for k in (10, 20, 30, 50)},
                }
                if report_ceilings
                else None
            ),
        },
        "n_conversations": n_conversations,
        "n_questions": agg["n_questions"],
        "per_category": agg["per_category"],
        "macro": agg["macro"],
        "micro": agg["micro"],
        "latency_ms": agg["latency_ms"],
        "mean_context_tokens": agg["mean_context_tokens"],
        "per_question": rows,
    }
    return result


# ---------------------------------------------------------------------------
# P9 step D: `--from-dump` -- evaluate a --dump-candidates output (optionally
# re-ranked by an offline LLM) WITHOUT touching memory_os_v4 at all.
# ---------------------------------------------------------------------------
def run_bench_from_dump(
    dataset: str,
    name: str,
    candidates_dir: str,
    llmrank_dir: Optional[str] = None,
    top_k: int = 10,
    limit: Optional[int] = None,
    locomo_path: Optional[str] = None,
    lme_path: Optional[str] = None,
    stratified: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Score a `--dump-candidates` output (P9 step D) exactly like `run_bench`
    would, but WITHOUT building a retriever, loading any model, or calling
    `add()`/`search()` -- purely reads `<candidates_dir>/<conv_id>.jsonl`
    (`bench.enrich.load_candidate_dump`) for each question's top-20 dumped
    candidates + `label_map`, optionally re-orders them with an offline
    LLM's ranking from `<llmrank_dir>/<conv_id>.jsonl`
    (`bench.enrich.load_llm_rankings`) via
    `bench.retrievers.apply_llm_ranking` -- the SAME combination function
    `V4LLMRerankedRetriever.search` uses, so this mode and a live
    `--retriever v4_llmrank` run can never disagree on how a ranking is
    applied -- then scores with the same `bench.metrics` functions
    `run_bench`'s own loop uses.

    `llmrank_dir=None` (or a `qid` missing from it) means "no LLM ranking
    yet" -- `apply_llm_ranking` then returns the dump's own order, so running
    this with no `llmrank_dir` reproduces the *ceiling* the matching
    `--dump-candidates` run already measured (its own top-`top_k` metrics),
    useful as a sanity check before any LLM ranking exists.

    `latency_ms` is 0.0 for every question -- genuinely no retrieval
    happened here -- flagged in `metadata["from_dump"]["note"]` so a reader
    of the results JSON never mistakes this for a timed run.
    """
    import random

    random.seed(seed)

    candidates_dir_path = Path(candidates_dir)
    llmrank_dir_path = Path(llmrank_dir) if llmrank_dir else None

    convs = list(
        load_dataset(dataset, limit, locomo_path=locomo_path, lme_path=lme_path, stratified=stratified)
    )

    rows: list[dict[str, Any]] = []
    n_conversations = 0
    n_questions_total = 0
    n_missing_dump = 0
    n_llm_ranked = 0
    n_fallback_no_llm_ranking = 0

    for conv in convs:
        n_conversations += 1
        dump_records = load_candidate_dump(candidates_dir_path, conv.conv_id)
        llm_rankings = load_llm_rankings(llmrank_dir_path, conv.conv_id) if llmrank_dir_path is not None else {}
        text_by_id = {item.id: item.text for item in conv.items}

        for q in conv.questions:
            n_questions_total += 1
            record = dump_records.get(q.qid)
            candidate_ids = record.turn_ids if record is not None else []
            if record is None:
                n_missing_dump += 1

            llm_ranked_ids: Optional[list[str]] = None
            labels = llm_rankings.get(q.qid) if record is not None else None
            if labels:
                mapped = [record.label_map[label] for label in labels if label in record.label_map]
                llm_ranked_ids = mapped or None

            if llm_ranked_ids:
                n_llm_ranked += 1
            else:
                n_fallback_no_llm_ranking += 1

            ranked = apply_llm_ranking(candidate_ids, llm_ranked_ids, top_k)
            top_texts = [text_by_id.get(i, "") for i in ranked]

            rows.append(
                {
                    "qid": q.qid,
                    "category": q.category,
                    "r5": recall_at_k(ranked, q.evidence_ids, 5),
                    "r10": recall_at_k(ranked, q.evidence_ids, 10),
                    "mrr": mrr(ranked, q.evidence_ids),
                    "ndcg10": ndcg_at_k(ranked, q.evidence_ids, 10),
                    "latency_ms": 0.0,
                    "context_tokens": context_tokens(top_texts),
                }
            )

    agg = aggregate(rows)

    import platform
    from datetime import datetime, timezone

    result = {
        "name": name,
        "dataset": dataset,
        "retriever": "v4_llmrank_from_dump",
        "config": {
            "top_k": top_k,
            "limit": limit,
            "seed": seed,
            "stratified": stratified,
            "config_overrides": {},
        },
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "from_dump": {
                "candidates_dir": str(candidates_dir_path),
                "llmrank_dir": str(llmrank_dir_path) if llmrank_dir_path is not None else None,
                "n_conversations": n_conversations,
                "n_questions": n_questions_total,
                "n_missing_dump": n_missing_dump,
                "n_llm_ranked": n_llm_ranked,
                "n_fallback_no_llm_ranking": n_fallback_no_llm_ranking,
                "fraction_llm_ranked": (n_llm_ranked / n_questions_total) if n_questions_total else 0.0,
                "note": (
                    "latency_ms is 0.0 for every question -- this run touched no retriever/model/"
                    "engine, it only re-scored a --dump-candidates output (see bench/README.md)."
                ),
            },
        },
        "n_conversations": n_conversations,
        "n_questions": agg["n_questions"],
        "per_category": agg["per_category"],
        "macro": agg["macro"],
        "micro": agg["micro"],
        "latency_ms": agg["latency_ms"],
        "mean_context_tokens": agg["mean_context_tokens"],
        "per_question": rows,
    }
    return result


# ---------------------------------------------------------------------------
# Persistence: versioned results JSON + RESULTS.md ledger row
# ---------------------------------------------------------------------------
def next_version_path(results_dir: Path, name: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (results_dir / f"{name}_v{n}.json").exists():
        n += 1
    return results_dir / f"{name}_v{n}.json"


def write_result_json(result: dict[str, Any], results_dir: Path) -> Path:
    path = next_version_path(results_dir, result["name"])
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _find_table_insert_point(lines: list[str]) -> int:
    """Return the line index right after the last existing table row.

    Looks for the markdown header-separator row (all `-`, `|`, ` ` chars)
    and returns the index just past the last contiguous `|`-prefixed row
    that follows it (so a brand-new table with no data rows yet inserts
    right after the separator, and a table with N rows inserts after row N).
    """
    sep_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.replace("|", "").replace(" ", "").replace(":", "") != "" and set(
            stripped.replace("|", "").replace(" ", "").replace(":", "")
        ) <= {"-"}:
            sep_idx = i
            break
    if sep_idx is None:
        raise RuntimeError("could not find the ledger's markdown table header separator row")
    insert_at = sep_idx + 1
    while insert_at < len(lines) and lines[insert_at].strip().startswith("|"):
        insert_at += 1
    return insert_at


def next_ledger_number(ledger_path: Path) -> int:
    if not ledger_path.exists():
        return 1
    max_n = 0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and cells[0].isdigit():
                max_n = max(max_n, int(cells[0]))
    return max_n + 1


def describe_pipeline(
    retriever_key: str, config_overrides: dict, plan_info: Optional[dict] = None
) -> dict[str, str]:
    """Map a retriever key (+ v4 config overrides) to the ledger's descriptive columns.

    `plan_info` (v4_planned only): the run's `metadata["planning"]` dict (see
    `run_bench`'s result), used to describe the BENCH-applied reranker/fusion
    (which live outside `config_overrides` -- see `V4PlannedRetriever`'s
    class docstring on why its reranker is independent of
    `memory_os_v4.Config.rerank_model`) instead of the generic v4 defaults.
    """
    if retriever_key == "v3equiv":
        return {"embedding": "all-MiniLM-L6-v2", "chunking": "128c/28ov (v3equiv, charitable)", "sparse": "-", "fusion": "-", "rerank": "-"}
    if retriever_key == "v3exact":
        return {"embedding": "all-MiniLM-L6-v2", "chunking": "128c/28ov (v3exact, literal, L2>0.8 drop)", "sparse": "-", "fusion": "-", "rerank": "-"}
    if retriever_key == "dense":
        return {"embedding": "all-MiniLM-L6-v2", "chunking": "turn (whole)", "sparse": "-", "fusion": "-", "rerank": "-"}
    if retriever_key == "bm25":
        return {"embedding": "-", "chunking": "turn (whole)", "sparse": "BM25(k1=1.2,b=0.75)", "fusion": "-", "rerank": "-"}
    if retriever_key in ("v4_planned", "v4_llmrank"):
        embed = config_overrides.get("embed_model", "default")
        use_bm25 = config_overrides.get("use_bm25", "default")
        plan_info = plan_info or {}
        plan_mode = plan_info.get("plan_mode", "?")
        fusion_mode = plan_info.get("fusion_mode", "rrf")
        fusion_label = "RRF" if fusion_mode == "rrf" else fusion_mode
        fusion = f"{fusion_label}(k={plan_info.get('rrf_k', 60)}, plan={plan_mode})"
        if plan_info.get("fact_links"):
            fusion += "+links"
        if plan_info.get("signal_confidence"):
            fusion += "+conf"
        if plan_info.get("iterative_budget"):
            fusion += "+iter"
        if plan_info.get("adjacency_pair"):
            fusion += "+adj"
        if plan_info.get("subquery_mmr"):
            fusion += f"+mmr({plan_info.get('mmr_lambda', 0.6)})"
        if plan_info.get("facet_channel"):
            fusion += f"+facets(w={plan_info.get('facet_weight', 0.5)})"
        if plan_info.get("implications"):
            fusion += "+impl"
        if retriever_key == "v4_llmrank":
            # P9 step D: reordered on top of the parent's own top-20 by an
            # offline LLM ranking (bench.retrievers.V4LLMRerankedRetriever) --
            # see the run's metadata["llm_rerank"] for coverage.
            fusion += "+llmrank"
        rerank_model = plan_info.get("rerank_model")
        rerank = "-"
        if rerank_model:
            blend = "blend0.5" if plan_info.get("plan_rerank_blend", True) else "ce-only"
            rerank = f"{rerank_model} (bench, top{plan_info.get('rerank_top_n', 30)}, {blend})"
        chunking = "turn + planned queries + enriched facts (v4 store)"
        return {
            "embedding": str(embed),
            "chunking": chunking,
            "sparse": f"BM25={use_bm25}",
            "fusion": fusion,
            "rerank": rerank,
        }
    if retriever_key == "v4_llmrank_from_dump":
        return {
            "embedding": "-",
            "chunking": "from --dump-candidates (no engine call)",
            "sparse": "-",
            "fusion": "llmrank-or-dump-order",
            "rerank": "-",
        }
    if retriever_key in ("v4", "v4_enriched"):
        embed = config_overrides.get("embed_model", "default")
        fusion = config_overrides.get("fusion", "default")
        rerank_model = config_overrides.get("rerank_model")
        use_bm25 = config_overrides.get("use_bm25", "default")
        rerank = rerank_model if rerank_model else ("off" if config_overrides.get("rerank") is False else "default")
        chunking = "turn (v4 store)"
        if retriever_key == "v4_enriched":
            chunking = "turn + enriched facts (v4 store)"
        return {
            "embedding": str(embed),
            "chunking": chunking,
            "sparse": f"BM25={use_bm25}",
            "fusion": str(fusion),
            "rerank": str(rerank),
        }
    return {"embedding": "?", "chunking": "?", "sparse": "?", "fusion": "?", "rerank": "?"}


def append_ledger_row(
    ledger_path: Path,
    result: dict[str, Any],
    result_json_path: Path,
    retriever_key: str,
    config_overrides: dict,
    notes: str = "",
    plan_info: Optional[dict] = None,
) -> str:
    pipeline = describe_pipeline(retriever_key, config_overrides, plan_info=plan_info)
    macro = result["macro"]
    p50 = result["latency_ms"]["p50"]
    row_cells = [
        str(next_ledger_number(ledger_path)),
        f"{result['name']} (`bench/results/{result_json_path.name}`)",
        pipeline["embedding"],
        pipeline["chunking"],
        pipeline["sparse"],
        pipeline["fusion"],
        pipeline["rerank"],
        f"{macro.get('r5', 0.0):.3f}",
        f"{macro.get('r10', 0.0):.3f}",
        f"{macro.get('mrr', 0.0):.3f}",
        f"{macro.get('ndcg10', 0.0):.3f}",
        f"{p50:.1f}",
        notes,
    ]
    lines = ledger_path.read_text(encoding="utf-8").splitlines() if ledger_path.exists() else []
    insert_at = _find_table_insert_point(lines)
    new_line = "| " + " | ".join(row_cells) + " |"
    lines.insert(insert_at, new_line)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return new_line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_summary(result: dict[str, Any]) -> None:
    macro = result["macro"]
    lat = result["latency_ms"]
    print(
        f"[{result['name']}] dataset={result['dataset']} retriever={result['retriever']} "
        f"n_conv={result['n_conversations']} n_q={result['n_questions']}"
    )
    print(
        f"  macro  R@5={macro.get('r5', 0):.3f}  R@10={macro.get('r10', 0):.3f}  "
        f"MRR={macro.get('mrr', 0):.3f}  nDCG@10={macro.get('ndcg10', 0):.3f}"
    )
    print(f"  latency p50={lat['p50']:.1f}ms  p95={lat['p95']:.1f}ms  mean={lat['mean']:.1f}ms")
    print("  per-category R@10:", {c: f"{v.get('r10', 0):.3f} (n={v.get('n')})" for c, v in result["per_category"].items()})
    ceilings = result["metadata"].get("ceilings")
    if ceilings:
        macro_c = ceilings["macro"]
        print(
            "  ceilings (macro, final ranked list): "
            f"R@10={macro_c['r10']:.3f}  R@20={macro_c['r20']:.3f}  "
            f"R@30={macro_c['r30']:.3f}  R@50={macro_c['r50']:.3f}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["locomo", "longmemeval_oracle"], required=True)
    p.add_argument(
        "--retriever",
        choices=["v3equiv", "v3exact", "dense", "bm25", "v4", "v4_enriched", "v4_planned", "v4_llmrank"],
        required=True,
        help="ignored (any value is accepted) when --from-dump is given -- see --from-dump below",
    )
    p.add_argument("--config-json", default="{}", help='v4 config overrides, e.g. \'{"fusion":"rrf"}\'')
    p.add_argument(
        "--enrich-dir",
        default=None,
        help=f"--retriever v4_enriched only: dir of <conv_id>.jsonl offline-enrichment files "
        f"(default: {DEFAULT_ENRICH_DIR})",
    )
    p.add_argument(
        "--enrich-mode",
        choices=["facts", "facts+keywords"],
        default="facts",
        help="--retriever v4_enriched only: 'facts' indexes extracted facts alongside raw turns; "
        "'facts+keywords' also appends extracted keywords to the raw turn's own indexed text",
    )
    p.add_argument(
        "--allow-partial-enrich",
        action="store_true",
        help="--retriever v4_enriched only: run even if --enrich-dir is missing files for some "
        "conversations (coverage is still recorded); without this flag, missing files fail fast",
    )
    p.add_argument("--name", required=True, help="base name for results/<name>_vN.json and the ledger row")
    p.add_argument("--limit", type=int, default=None, help="first N conversations (locomo) or questions (longmemeval_oracle)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    p.add_argument("--locomo-path", default=None)
    p.add_argument("--lme-path", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--no-ledger", action="store_true", help="skip appending to RESULTS.md (used by tests)")
    p.add_argument(
        "--stratified",
        action="store_true",
        help="round-robin --limit across question_type/category instead of a file-order prefix "
        "(longmemeval_oracle only; no-op for locomo, see E7 docs/ABLATIONS.md)",
    )
    # ---- P9 §A-C: --retriever v4_planned only ----
    p.add_argument(
        "--queries-dir",
        default=None,
        help=f"--retriever v4_planned only: dir of <conv_id>.jsonl offline query-plan files "
        f"(default: {DEFAULT_QUERIES_DIR})",
    )
    p.add_argument(
        "--plan-mode",
        choices=["sub", "sub+hyp", "hyp"],
        default="sub+hyp",
        help="--retriever v4_planned only: which parts of the offline query plan to fuse in "
        "alongside the original question (P9 §A)",
    )
    p.add_argument("--plan-weight-original", type=float, default=1.0)
    p.add_argument("--plan-weight-sub", type=float, default=1.0)
    p.add_argument("--plan-weight-hyp", type=float, default=0.7)
    p.add_argument("--rrf-k", type=int, default=60, help="--retriever v4_planned only: RRF constant")
    p.add_argument(
        "--allow-partial-plan",
        action="store_true",
        help="--retriever v4_planned only: run even if --queries-dir is missing files for some "
        "conversations (every question in a missing conversation falls back to single-query "
        "search); without this flag, missing files fail fast",
    )
    p.add_argument(
        "--rerank-model",
        default=None,
        help="--retriever v4_planned only: cross-encoder model the BENCH applies itself to the top "
        "--rerank-top-n fused turns (P9 §C), independent of the v4 engine's own (disabled) reranker "
        "-- e.g. tomaarsen/ms-marco-ettin-17m-reranker or the local -68m variant",
    )
    p.add_argument("--rerank-device", default="auto")
    p.add_argument("--rerank-top-n", type=int, default=30)
    plan_rerank_blend_group = p.add_mutually_exclusive_group()
    plan_rerank_blend_group.add_argument(
        "--plan-rerank-blend",
        dest="plan_rerank_blend",
        action="store_true",
        default=True,
        help="--retriever v4_planned only (default): blend the reranked top-n 0.5/0.5 with the "
        "pre-rerank fused score, like memory_os_v4.retrieval's own rerank blend",
    )
    plan_rerank_blend_group.add_argument(
        "--plan-rerank-ce-only",
        dest="plan_rerank_blend",
        action="store_false",
        help="--retriever v4_planned only: rank the reranked top-n by the cross-encoder score alone",
    )
    p.add_argument(
        "--fact-links",
        action="store_true",
        help="--retriever v4_planned only: P9 §B A-Mem-style fact link expansion",
    )
    p.add_argument("--link-seed", type=int, default=8)
    p.add_argument("--link-decay", type=float, default=0.5)
    # ---- Idées inter-disciplines (docs/IDEAS.md, 2026-09-14): --retriever v4_planned only ----
    p.add_argument(
        "--fusion-mode",
        choices=["rrf", "quota", "quota_then_rrf"],
        default="rrf",
        help="--retriever v4_planned only: idea 1 (Sainte-Laguë proportional fusion). "
        "'quota' allocates the whole --rerank-top-n by Sainte-Laguë seats across "
        "query lists (party weights = --plan-weight-original/-sub/-hyp), RRF order "
        "for anything left over; 'quota_then_rrf' allocates only the first "
        "--quota-then-rrf-boundary positions by quota, plain RRF after",
    )
    p.add_argument(
        "--quota-then-rrf-boundary",
        type=int,
        default=10,
        help="--fusion-mode quota_then_rrf only: how many leading positions the "
        "Sainte-Laguë quota decides before falling back to plain RRF order",
    )
    p.add_argument(
        "--subquery-mmr",
        action="store_true",
        help="--retriever v4_planned only: idea 2, greedy MMR-style re-ordering that "
        "penalises a candidate by --mmr-lambda**n once its owner query list (the "
        "list where it ranks best) is already represented n times",
    )
    p.add_argument("--mmr-lambda", type=float, default=0.6)
    p.add_argument(
        "--signal-confidence",
        action="store_true",
        help="--retriever v4_planned only: idea 3, scale each query list's RRF "
        "weight by the normalised top1/top2 margin of its own raw engine scores "
        "(clipped to [0.3, 1.0]) -- a 'flat' list is down-weighted, never boosted",
    )
    p.add_argument(
        "--adjacency-pair",
        action="store_true",
        help="--retriever v4_planned only: idea 4, for each top-10 fused turn that "
        "reads as a question asked by a different speaker than the immediately "
        "following same-session turn, add that following turn at score*0.8",
    )
    p.add_argument(
        "--iterative-budget",
        action="store_true",
        help="--retriever v4_planned only: idea 5 (optimal foraging), re-run any "
        "sub-query with none of its own hits in the fused top-20 alone at top_k*8 "
        "and merge its own top-5 at fused positions 11-20, capped at 2 extra "
        "searches per question",
    )
    p.add_argument(
        "--report-ceilings",
        action="store_true",
        help="Compute recall@k for k in (10, 20, 30, 50) of the FINAL ranked list "
        "from a single search(top_k=50) call per question (evidence is used only "
        "to SCORE this already-computed list, never fed back into retrieval); "
        "written to metadata.ceilings and printed in the run summary",
    )
    # ---- Idea 8 (docs/IDEAS.md, 2026-09-19): --retriever v4_planned only ----
    p.add_argument(
        "--facet-channel",
        action="store_true",
        help="--retriever v4_planned only: idea 8 facet channel. Builds a "
        "closed-predicate (subject, predicate) -> turn-id index per "
        "conversation from --facets-dir and, per question, adds the union "
        "of turns matching its own (subject, predicate) facets (--qfacets-dir) "
        "as one more RRF fusion party at --facet-weight, ranked by each "
        "pool member's own best rank across the original/sub-query/hypothesis "
        "hit lists. Exact predicate match only (no family widening). A "
        "question with no facets, or a conversation with no facet file, is "
        "unchanged.",
    )
    p.add_argument(
        "--facet-weight",
        type=float,
        default=0.5,
        help="--facet-channel only: RRF party weight for the facet pool (default 0.5). "
        "0 (or no facets for a question) leaves the fusion byte-identical to "
        "--facet-channel being off.",
    )
    p.add_argument(
        "--facets-dir",
        default=None,
        help=f"--facet-channel/--implications only: dir of <conv_id>.jsonl extracted-fact "
        f"facet+implication files (default: {DEFAULT_FACETS_DIR})",
    )
    p.add_argument(
        "--qfacets-dir",
        default=None,
        help=f"--facet-channel only: dir of <conv_id>.jsonl question-facet files "
        f"(default: {DEFAULT_QFACETS_DIR})",
    )
    p.add_argument(
        "--implications",
        action="store_true",
        help="--retriever v4_planned only: idea 8 implications channel, independent "
        "of --facet-channel. Indexes each extracted fact's implications (--facets-dir) "
        "as additional add_facts items (parent_ids=[turn_id], tags=['implication'], "
        "importance=0.5) so a hit on an implication is credited to its parent turn.",
    )
    # ---- P9 step D: candidate dumps + offline LLM listwise reranking ----
    p.add_argument(
        "--dump-candidates",
        default=None,
        metavar="DIR",
        help="--retriever v4_planned (or v4_llmrank) only: write one "
        "<conv_id>.jsonl file per conversation to DIR, one line per question, "
        "recording the retriever's FINAL top-20 ranking for that question "
        "(P9 step D, see bench/README.md) -- for an offline LLM listwise-"
        "reranking worker to read; never includes evidence ids/answers/categories. "
        "Does not change --top-k's normal metrics (a single oversampled search() "
        "call serves both).",
    )
    p.add_argument(
        "--llmrank-dir",
        default=None,
        metavar="DIR",
        help="--retriever v4_llmrank (or --from-dump) only: dir of <conv_id>.jsonl "
        'offline LLM ranking files ({"qid", "ranked": ["c07", "c02", ...]}, labels '
        "from a matching --candidates-dir dump) produced by other worker sessions; "
        "a qid with no ranking here falls back to the parent/dump order",
    )
    p.add_argument(
        "--candidates-dir",
        default=None,
        metavar="DIR",
        help="--retriever v4_llmrank (or --from-dump) only: dir of <conv_id>.jsonl "
        "--dump-candidates output (P9 step D) this run reads its top-20 candidates "
        "and label->turn_id mapping from",
    )
    p.add_argument(
        "--from-dump",
        action="store_true",
        help="Evaluate directly from --candidates-dir (+ optional --llmrank-dir) "
        "WITHOUT building a retriever or touching memory_os_v4 at all -- fast, "
        "pure re-scoring of an existing dump (P9 step D). --retriever is ignored; "
        "--candidates-dir is required. latency_ms is reported as 0.0 for every "
        "question (see metadata.from_dump.note in the written results JSON).",
    )
    args = p.parse_args(argv)

    if args.from_dump and not args.candidates_dir:
        print("--from-dump requires --candidates-dir", file=sys.stderr)
        return 2

    try:
        config_overrides = json.loads(args.config_json)
    except json.JSONDecodeError as exc:
        print(f"--config-json is not valid JSON: {exc}", file=sys.stderr)
        return 2

    if args.stratified and args.dataset != "longmemeval_oracle":
        print(
            f"[warn] --stratified is a no-op for --dataset {args.dataset!r} "
            "(only longmemeval_oracle has one question per conversation)",
            file=sys.stderr,
        )

    try:
        if args.from_dump:
            result = run_bench_from_dump(
                dataset=args.dataset,
                name=args.name,
                candidates_dir=args.candidates_dir,
                llmrank_dir=args.llmrank_dir,
                top_k=args.top_k,
                limit=args.limit,
                locomo_path=args.locomo_path,
                lme_path=args.lme_path,
                stratified=args.stratified,
                seed=args.seed,
            )
        else:
            result = run_bench(
                dataset=args.dataset,
                retriever_key=args.retriever,
                name=args.name,
                top_k=args.top_k,
                limit=args.limit,
                config_overrides=config_overrides,
                seed=args.seed,
                locomo_path=args.locomo_path,
                lme_path=args.lme_path,
                stratified=args.stratified,
                enrich_dir=args.enrich_dir,
                enrich_mode=args.enrich_mode,
                allow_partial_enrich=args.allow_partial_enrich,
                queries_dir=args.queries_dir,
                plan_mode=args.plan_mode,
                weight_original=args.plan_weight_original,
                weight_sub=args.plan_weight_sub,
                weight_hyp=args.plan_weight_hyp,
                rrf_k=args.rrf_k,
                rerank_model=args.rerank_model,
                rerank_device=args.rerank_device,
                rerank_top_n=args.rerank_top_n,
                plan_rerank_blend=args.plan_rerank_blend,
                fact_links=args.fact_links,
                link_seed=args.link_seed,
                link_decay=args.link_decay,
                allow_partial_plan=args.allow_partial_plan,
                llmrank_dir=args.llmrank_dir,
                candidates_dir=args.candidates_dir,
                dump_candidates_dir=args.dump_candidates,
                fusion_mode=args.fusion_mode,
                quota_then_rrf_boundary=args.quota_then_rrf_boundary,
                subquery_mmr=args.subquery_mmr,
                mmr_lambda=args.mmr_lambda,
                signal_confidence=args.signal_confidence,
                adjacency_pair=args.adjacency_pair,
                iterative_budget=args.iterative_budget,
                report_ceilings=args.report_ceilings,
                facet_channel=args.facet_channel,
                facet_weight=args.facet_weight,
                facets_dir=args.facets_dir,
                qfacets_dir=args.qfacets_dir,
                implications=args.implications,
            )
    except ImportError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    except RuntimeError as exc:
        # e.g. V4EnrichedRetriever fail-fast on missing --enrich-dir files.
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    results_dir = Path(args.results_dir)
    result_json_path = write_result_json(result, results_dir)
    print(f"wrote {result_json_path}")

    if not args.no_ledger:
        ledger_path = Path(args.ledger)
        notes = args.notes or f"{args.dataset}, n={result['n_questions']}, top_k={args.top_k}"
        enrichment_meta = result["metadata"].get("enrichment")
        if enrichment_meta:
            notes = (
                f"{notes}; enrich_mode={enrichment_meta['enrich_mode']}, "
                f"coverage={enrichment_meta['coverage_turns']:.2f} "
                f"({enrichment_meta['turns_with_facts']}/{enrichment_meta['turns_total']} turns), "
                f"facts={enrichment_meta['facts_added']}, "
                f"convs_missing_file={enrichment_meta['n_conversations_missing_file']}"
            )
            search_stats = result["metadata"].get("search_stats") or {}
            n_searches = search_stats.get("n_searches", 0)
            if n_searches:
                notes += (
                    f", top_hit facts/episodes/other="
                    f"{search_stats.get('top_hit_facts', 0)}/"
                    f"{search_stats.get('top_hit_episodes', 0)}/"
                    f"{search_stats.get('top_hit_other', 0)} (n_searches={n_searches})"
                )
        planning_meta = result["metadata"].get("planning")
        if planning_meta:
            notes = (
                f"{notes}; plan_mode={planning_meta['plan_mode']}, "
                f"fraction_with_plan={planning_meta['fraction_questions_with_plan']:.2f} "
                f"({planning_meta['n_plan_used']}/{planning_meta['n_searches']} questions), "
                f"fact_links={planning_meta['fact_links']}, "
                f"rerank_top_n={planning_meta['rerank_top_n']}, "
                f"plan_rerank_blend={planning_meta['plan_rerank_blend']}"
            )
        facets_meta = result["metadata"].get("facets")
        if facets_meta and (facets_meta["facet_channel"] or facets_meta["implications"]):
            notes = (
                f"{notes}; facet_channel={facets_meta['facet_channel']} "
                f"(weight={facets_meta['facet_weight']}, "
                f"fraction_with_facets={facets_meta['fraction_questions_with_facets']:.2f}, "
                f"mean_pool_size={facets_meta['mean_pool_size']:.1f}, "
                f"mean_new_vs_top50={facets_meta['mean_new_vs_fused_top50']:.1f}), "
                f"implications={facets_meta['implications']} "
                f"(n_indexed={facets_meta['n_implications_indexed']})"
            )
        llm_rerank_meta = result["metadata"].get("llm_rerank")
        if llm_rerank_meta:
            notes = (
                f"{notes}; llm_rerank fraction_llm_ranked={llm_rerank_meta['fraction_llm_ranked']:.2f} "
                f"({llm_rerank_meta['llm_ranked_questions']}/{llm_rerank_meta['n_searches']} questions), "
                f"llmrank_dir={llm_rerank_meta['llmrank_dir']}, candidates_dir={llm_rerank_meta['candidates_dir']}"
            )
        dump_meta = result["metadata"].get("dump_candidates")
        if dump_meta:
            notes = (
                f"{notes}; dump_candidates dir={dump_meta['dir']}, "
                f"n_files={dump_meta['n_files']}, n_lines={dump_meta['n_lines']}, "
                f"max_candidates={dump_meta['max_candidates']}"
            )
        from_dump_meta = result["metadata"].get("from_dump")
        if from_dump_meta:
            notes = (
                f"{notes}; from_dump candidates_dir={from_dump_meta['candidates_dir']}, "
                f"llmrank_dir={from_dump_meta['llmrank_dir']}, "
                f"fraction_llm_ranked={from_dump_meta['fraction_llm_ranked']:.2f} "
                f"({from_dump_meta['n_llm_ranked']}/{from_dump_meta['n_questions']} questions), "
                f"n_missing_dump={from_dump_meta['n_missing_dump']}"
            )
        ceilings_meta = result["metadata"].get("ceilings")
        if ceilings_meta:
            mc = ceilings_meta["macro"]
            notes = (
                f"{notes}; ceilings(macro) R@10={mc['r10']:.3f} R@20={mc['r20']:.3f} "
                f"R@30={mc['r30']:.3f} R@50={mc['r50']:.3f}"
            )
        # describe_pipeline's fusion label reads facet_channel/implications
        # off `plan_info` too (idea 8, docs/IDEAS.md) -- merge them in rather
        # than widen describe_pipeline's signature for one more dict.
        combined_plan_info = dict(planning_meta or {})
        if facets_meta:
            combined_plan_info.update(
                {
                    "facet_channel": facets_meta["facet_channel"],
                    "facet_weight": facets_meta["facet_weight"],
                    "implications": facets_meta["implications"],
                }
            )
        row = append_ledger_row(
            ledger_path,
            result,
            result_json_path,
            result["retriever"],
            config_overrides,
            notes=notes,
            plan_info=combined_plan_info,
        )
        print(f"appended to {ledger_path}:\n  {row}")

    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
