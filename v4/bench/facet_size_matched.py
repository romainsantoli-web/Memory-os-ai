"""Size-matched control for the facet index (docs/IDEAS.md, idea 8). Model-free, writes nothing.

For every question, the facet pool adds k new turns to the plain retriever's top 50. The fair
control is the SAME retriever taken k candidates deeper: recall of gold evidence in
(top50 U facet_pool) versus recall in top(50+k), per question, from the 150-deep no-reranker dump.
Gold evidence is used for evaluation only.
Usage: PYTHONPATH=src:. python -m bench.facet_size_matched
"""
import collections, json, random, re
from bench.locomo import load_locomo

E = "bench/data/enrich"
DEEP = "bench/data/dumps/locomo_v4_plan_norerank_k150"; K50 = "bench/data/dumps/locomo_v4_plan_best_k50"
NAMES = {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop"}

def families():
    fam = {}
    for line in open(f"{E}/PREDICATES.md"):
        m = re.match(r"^([a-z/]+):\s*(.+)$", line.strip())
        if m and m.group(1) != "fallback":
            for p in m.group(2).split(","): fam[p.strip()] = m.group(1)
    members = collections.defaultdict(set)
    for p, f in fam.items(): members[f].add(p)
    return fam, members

def boot_ci(deltas, n=2000, seed=0):
    rnd = random.Random(seed); m = len(deltas); means = []
    for _ in range(n):
        means.append(sum(deltas[rnd.randrange(m)] for _ in range(m)) / m)
    means.sort(); return means[int(.025 * n)], means[int(.975 * n)]

def main():
    fam, members = families()
    convs = {c.conv_id: c for c in load_locomo("bench/data/locomo10.json")}
    deep, k50 = {}, {}
    for cid in convs:
        for l in open(f"{DEEP}/{cid}.jsonl"): d = json.loads(l); deep[d["qid"]] = [c["turn_id"] for c in d["candidates"]]
        for l in open(f"{K50}/{cid}.jsonl"): d = json.loads(l); k50[d["qid"]] = [c["turn_id"] for c in d["candidates"]]
    same = sum(1 for q in deep if set(deep[q][:50]) == set(k50[q])); jac = sum(len(set(deep[q][:50]) & set(k50[q])) / len(set(deep[q][:50]) | set(k50[q])) for q in deep) / len(deep)
    print(f"top-50 set of the new dump vs the earlier k50 dump: identical for {same}/{len(deep)} questions, mean Jaccard {jac:.3f}")
    for mode in ("exact", "family"):
        rows = collections.defaultdict(list); capped = 0; ks = []
        for cid, conv in convs.items():
            idx = collections.defaultdict(set)
            for l in open(f"{E}/locomo_facets/{cid}.jsonl"):
                d = json.loads(l); turn = d["fid"].split("#")[0]
                for f in d["facets"]:
                    if f["predicate"] != "other": idx[(f["subject"], f["predicate"])].add(turn)
            qf = {json.loads(l)["qid"]: json.loads(l)["facets"] for l in open(f"{E}/locomo_qfacets/{cid}.jsonl")}
            for q in conv.questions:
                gold = set(q.evidence_ids); ranked = deep[q.qid]; top = set(ranked[:50]); pool = set()
                for f in qf[q.qid]:
                    preds = {f["predicate"]} if mode == "exact" else members.get(fam.get(f["predicate"], ""), {f["predicate"]})
                    for p in preds: pool |= idx[(f["subject"], p)]
                k = len(pool - top); ks.append(k)
                if 50 + k > len(ranked): capped += 1
                base = len(gold & top) / len(gold); fac = len(gold & (top | pool)) / len(gold); ctl = len(gold & set(ranked[:50 + k])) / len(gold)
                rows[str(q.category)].append((base, fac, ctl))
        print(f"\n[{mode}] new candidates per question: median {sorted(ks)[len(ks)//2]}, mean {sum(ks)/len(ks):.1f} | questions where 50+k exceeds dump depth (control capped, favours facets): {capped}")
        print(f"    {'category':11s} {'n':>4s} {'top-50':>7s} {'facets':>7s} {'control':>8s} {'facets-control':>15s}  95% CI (bootstrap)   facets>ctl / <ctl")
        macro = [0, 0, 0]
        for cat in sorted(rows):
            r = rows[cat]; n = len(r); b = sum(x[0] for x in r) / n; f = sum(x[1] for x in r) / n; c = sum(x[2] for x in r) / n
            lo, hi = boot_ci([x[1] - x[2] for x in r]); better = sum(1 for x in r if x[1] > x[2]); worse = sum(1 for x in r if x[1] < x[2])
            macro = [macro[0] + b, macro[1] + f, macro[2] + c]
            print(f"    {NAMES[cat]:11s} {n:4d} {b:7.3f} {f:7.3f} {c:8.3f} {f-c:+15.3f}  [{lo:+.3f}, {hi:+.3f}]      {better:3d} / {worse:3d}")
        m = [x / len(rows) for x in macro]; print(f"    {'MACRO':11s} {'':4s} {m[0]:7.3f} {m[1]:7.3f} {m[2]:8.3f} {m[1]-m[2]:+15.3f}")

if __name__ == "__main__":
    main()
