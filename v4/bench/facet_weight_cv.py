"""Leave-one-conversation-out selection of the facet channel weight (V5.0).

The facet weight (0.5 / 1.0 / 2.0) was chosen on the very LoCoMo questions
it is reported on (docs/IDEAS.md, 2026-09-19). This script removes that
optimism without any new retrieval run: for each held-out conversation the
weight is picked on the nine OTHER conversations (best macro R@10, ties ->
smaller weight), then scored on the held-out one. Model-free: reads only the
`per_question` records of existing `bench/results/*.json` files.

    PYTHONPATH=src:. python -m bench.facet_weight_cv
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
RUNS = {
    0.0: "locomo_v4_plan_norerank_k150_dump_v1.json",
    0.5: "locomo_v5_facets_w05_v1.json",
    1.0: "locomo_v5_facets_w10_v1.json",
    2.0: "locomo_v5_facets_w20_v1.json",
}
METRICS = ("r10", "r5", "mrr")


def load(name: str) -> dict[str, dict]:
    data = json.loads((RESULTS / name).read_text())
    return {q["qid"]: q for q in data["per_question"]}


def conv_of(qid: str) -> str:
    return qid.split(":", 1)[0]


def macro(rows: list[dict], metric: str) -> float:
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r[metric])
    return sum(sum(v) / len(v) for v in by_cat.values()) / len(by_cat)


def main() -> None:
    runs = {w: load(name) for w, name in RUNS.items()}
    qids = sorted(runs[0.0])
    assert all(sorted(r) == qids for r in runs.values()), "runs cover different questions"
    convs = sorted({conv_of(q) for q in qids})

    print("weight  macro R@10 (all 10 conversations, in-sample)")
    for w in RUNS:
        print(f"  {w:>4}  {macro(list(runs[w].values()), 'r10'):.4f}")

    # Leave-one-conversation-out: pick on 9, score on 1.
    chosen: dict[str, float] = {}
    cv_rows: list[dict] = []
    base_rows: list[dict] = []
    for held in convs:
        train = [q for q in qids if conv_of(q) != held]
        best_w = max(RUNS, key=lambda w: (round(macro([runs[w][q] for q in train], "r10"), 6), -w))
        chosen[held] = best_w
        for q in qids:
            if conv_of(q) == held:
                cv_rows.append(runs[best_w][q])
                base_rows.append(runs[0.0][q])
    print("\nweight chosen per held-out conversation:", chosen)
    for m in METRICS:
        print(f"  cross-validated macro {m}: {macro(cv_rows, m):.4f}   baseline: {macro(base_rows, m):.4f}"
              f"   delta: {macro(cv_rows, m) - macro(base_rows, m):+.4f}")

    # Per-conversation sign + bootstrap over conversations (the independent unit).
    per_conv = {}
    for c in convs:
        cv = [r for r in cv_rows if conv_of(r["qid"]) == c]
        ba = [r for r in base_rows if conv_of(r["qid"]) == c]
        per_conv[c] = macro(cv, "r10") - macro(ba, "r10")
    print("\nper-conversation delta macro R@10 (cross-validated weight vs baseline):")
    for c, d in per_conv.items():
        print(f"  {c}: {d:+.4f}")
    print(f"  positive in {sum(d > 0 for d in per_conv.values())}/{len(per_conv)} conversations")

    rng = random.Random(0)
    deltas = []
    for _ in range(5000):
        sample = [rng.choice(convs) for _ in convs]
        cv = [r for c in sample for r in cv_rows if conv_of(r["qid"]) == c]
        ba = [r for c in sample for r in base_rows if conv_of(r["qid"]) == c]
        deltas.append(macro(cv, "r10") - macro(ba, "r10"))
    deltas.sort()
    print(f"  bootstrap over conversations, 95% CI: [{deltas[125]:+.4f}, {deltas[4874]:+.4f}]")

    print("\nper category, cross-validated vs baseline (R@10):")
    for cat in sorted({r["category"] for r in cv_rows}):
        cv = [r["r10"] for r in cv_rows if r["category"] == cat]
        ba = [r["r10"] for r in base_rows if r["category"] == cat]
        print(f"  cat {cat} (n={len(cv)}): {sum(cv)/len(cv):.4f} vs {sum(ba)/len(ba):.4f}  ({sum(cv)/len(cv)-sum(ba)/len(ba):+.4f})")


if __name__ == "__main__":
    main()
