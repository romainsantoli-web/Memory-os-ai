"""Second benchmark for the facet channel: LongMemEval oracle (stratified 150),
through the PRODUCTION path (`MemoryV4.search(facets=...)`, `facet_channel`
flag). Protocol pre-registered in docs/IDEAS.md ("second banc : protocole
pré-enregistré LongMemEval") BEFORE this script was first run: weight fixed
at 2.0 (chosen on LoCoMo), defaults everywhere else, no reranker.

Both arms are read from the SAME index, in the same process: the facet-less
arm is `search(..., facets=None)`, which the port guarantees is the plain
single-query path (tests/test_facets.py, flag-off/no-facets byte-identity).

    PYTHONPATH=src:. python -m bench.lme_facets_eval --mode pooled
    PYTHONPATH=src:. python -m bench.lme_facets_eval --mode oracle

Modes
- pooled : every session of the 150 questions in ONE index (the hard,
  production-like case: subject is always "user", pools are large).
- oracle : one index per question (~22 turns), the standard LME-oracle setup
  -- guardrail only (the channel can merely re-rank there).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from bench.metrics import mrr, recall_at_k
from bench.run_bench import load_dataset

ENRICH = Path(__file__).parent / "data" / "enrich"
RESULTS = Path(__file__).parent / "results"
LIMIT = 150
TOP_K = 10


def load_turn_facets() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted((ENRICH / "lme_facets").glob("batch_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["tid"]] = rec["facets"]
    return out


def load_question_facets() -> dict[str, list[dict]]:
    path = ENRICH / "lme_qfacets" / "questions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["qid"]: r["facets"] for r in rows}


def make_memory(tmpdir: str, weight: float, share: float | None):
    from memory_os_v4 import MemoryV4
    from memory_os_v4.config import Config

    return MemoryV4(
        data_dir=tmpdir,
        config=Config(facet_channel=True, facet_weight=weight, facet_share=share),
    )


def add_items(memory, items, turn_facets: dict[str, list[dict]]) -> None:
    episodes = []
    for item in items:
        prefix = f"{item.speaker}: " if item.speaker else ""
        episodes.append(
            {
                "text": f"{prefix}{item.text}",
                "role": "user",
                "project": "bench",
                "event_time": item.event_time,
                "item_id": item.id,
                "doc_ref": item.session_id,
                "facets": turn_facets.get(item.id) or [],
            }
        )
    memory.add_episodes_bulk(episodes)


def macro(rows: list[dict], key: str) -> float:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r[key])
    return sum(sum(v) / len(v) for v in by_cat.values()) / len(by_cat)


def score(ranked: list[str], evidence: list[str]) -> dict[str, float]:
    return {
        "r5": recall_at_k(ranked, evidence, 5),
        "r10": recall_at_k(ranked, evidence, 10),
        "mrr": mrr(ranked, evidence),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pooled", "oracle"], required=True)
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument(
        "--share", type=float, default=None,
        help="facet weight as a share of the total vote (config.facet_share); omit = absolute --weight",
    )
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    convs = list(load_dataset("longmemeval_oracle", LIMIT, stratified=True))
    turn_facets = load_turn_facets()
    qfacets = load_question_facets()
    rows: list[dict] = []
    t0 = time.time()

    def run_questions(memory, questions) -> None:
        for q in questions:
            facets = qfacets.get(q.qid) or []
            base = [h.item.id for h in memory.search(q.text, top_k=TOP_K, at_time=q.query_time)]
            t1 = time.time()
            hits = memory.search(q.text, top_k=TOP_K, at_time=q.query_time, facets=facets or None)
            lat = (time.time() - t1) * 1000
            fac = [h.item.id for h in hits]
            row = {"qid": q.qid, "category": q.category, "n_qfacets": len(facets),
                   "n_facet_hits_top10": sum(1 for h in hits if h.signals.get("facet")),
                   "latency_facets_ms": lat}
            row.update({f"base_{k}": v for k, v in score(base, q.evidence_ids).items()})
            row.update({f"facet_{k}": v for k, v in score(fac, q.evidence_ids).items()})
            rows.append(row)

    if args.mode == "pooled":
        tmp = tempfile.mkdtemp(prefix="lme_facets_pooled_")
        try:
            memory = make_memory(tmp, args.weight, args.share)
            seen: set[str] = set()
            items = []
            for conv in convs:
                for item in conv.items:
                    if item.id not in seen:
                        seen.add(item.id)
                        items.append(item)
            add_items(memory, items, turn_facets)
            print(f"[pooled] indexed {len(items)} turns in {time.time() - t0:.0f}s", flush=True)
            run_questions(memory, [q for conv in convs for q in conv.questions])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        for i, conv in enumerate(convs):
            tmp = tempfile.mkdtemp(prefix="lme_facets_oracle_")
            try:
                memory = make_memory(tmp, args.weight, args.share)
                add_items(memory, conv.items, turn_facets)
                run_questions(memory, conv.questions)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if (i + 1) % 25 == 0:
                print(f"[oracle] {i + 1}/{len(convs)} questions, {time.time() - t0:.0f}s", flush=True)

    summary = {
        "name": args.name or f"lme_v5_facets_{args.mode}_v1",
        "mode": args.mode, "weight": args.weight, "share": args.share, "n_questions": len(rows),
        "n_turns_with_facets": len(turn_facets),
        "n_questions_with_facets": sum(1 for r in rows if r["n_qfacets"]),
        "elapsed_s": time.time() - t0,
    }
    for arm in ("base", "facet"):
        for k in ("r5", "r10", "mrr"):
            summary[f"{arm}_macro_{k}"] = macro(rows, f"{arm}_{k}")
    summary["delta_macro_r10"] = summary["facet_macro_r10"] - summary["base_macro_r10"]

    rng = random.Random(0)
    deltas = []
    for _ in range(5000):
        sample = [rng.choice(rows) for _ in rows]
        if len({r["category"] for r in sample}) < len({r["category"] for r in rows}):
            continue
        deltas.append(macro(sample, "facet_r10") - macro(sample, "base_r10"))
    deltas.sort()
    summary["delta_ci95"] = [deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]]

    per_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        per_cat[cat] = {"n": len(sub),
                        "base_r10": sum(r["base_r10"] for r in sub) / len(sub),
                        "facet_r10": sum(r["facet_r10"] for r in sub) / len(sub)}
    summary["per_category"] = per_cat
    summary["n_better"] = sum(1 for r in rows if r["facet_r10"] > r["base_r10"])
    summary["n_worse"] = sum(1 for r in rows if r["facet_r10"] < r["base_r10"])
    lat = sorted(r["latency_facets_ms"] for r in rows)
    summary["latency_facets_p50_ms"] = lat[len(lat) // 2]

    out = RESULTS / f"{summary['name']}.json"
    if out.exists():
        raise SystemExit(f"{out} exists -- bump the version in --name, never overwrite")
    out.write_text(json.dumps({**summary, "per_question": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
