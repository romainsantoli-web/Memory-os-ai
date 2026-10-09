"""Offline, model-free analysis of the facet index (docs/IDEAS.md, idea 8).

For each question, builds the pool of turns whose annotated facts share a
(subject, predicate) with the question's facets, under three widths (exact
predicate / predicate family / every turn about that subject = size control),
and reports how much gold evidence missing from the 50-candidate dump it
recovers, against how many new candidates it injects. Gold evidence is used
for EVALUATION only. Reads JSON files; loads no model; writes nothing.
Usage: PYTHONPATH=src:. python -m bench.facet_ceiling_analysis
"""
import collections, json, re, statistics
from bench.locomo import load_locomo

E = "bench/data/enrich"; D = "bench/data/dumps/locomo_v4_plan_best_k50"
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

def main():
    fam, members = families()
    convs = {c.conv_id: c for c in load_locomo("bench/data/locomo10.json")}
    for mode in ("exact", "family", "subject"):
        agg = collections.defaultdict(collections.Counter); added = []
        for cid, conv in convs.items():
            idx = collections.defaultdict(set); by_subject = collections.defaultdict(set)
            for l in open(f"{E}/locomo_facets/{cid}.jsonl"):
                d = json.loads(l); turn = d["fid"].split("#")[0]
                for f in d["facets"]:
                    if f["predicate"] != "other":
                        idx[(f["subject"], f["predicate"])].add(turn); by_subject[f["subject"]].add(turn)
            qf = {json.loads(l)["qid"]: json.loads(l)["facets"] for l in open(f"{E}/locomo_qfacets/{cid}.jsonl")}
            dump = {json.loads(l)["qid"]: [c["turn_id"] for c in json.loads(l)["candidates"]] for l in open(f"{D}/{cid}.jsonl")}
            for q in conv.questions:
                gold = set(q.evidence_ids); pool = set()
                for f in qf[q.qid]:
                    if mode == "subject": pool |= by_subject[f["subject"]]
                    else:
                        preds = {f["predicate"]} if mode == "exact" else members.get(fam.get(f["predicate"], ""), {f["predicate"]})
                        for p in preds: pool |= idx[(f["subject"], p)]
                top = set(dump[q.qid]); lost = gold - top; a = agg[str(q.category)]
                a["n"] += 1; a["lost"] += len(lost); a["rec"] += len(lost & pool)
                a["c0"] += len(gold & top) / len(gold); a["c1"] += len(gold & (top | pool)) / len(gold)
                added.append(len(pool - top))
        m0 = sum(a["c0"] / a["n"] for a in agg.values()) / len(agg); m1 = sum(a["c1"] / a["n"] for a in agg.values()) / len(agg)
        print(f"\n[{mode}] lost recovered {sum(a['rec'] for a in agg.values())}/{sum(a['lost'] for a in agg.values())} | macro ceiling {m0:.3f} -> {m1:.3f} | new candidates/question: median {statistics.median(added):.0f}, p90 {sorted(added)[int(.9*len(added))]}, max {max(added)}")
        for cat in sorted(agg):
            a = agg[cat]; print(f"    {NAMES[cat]:11s} n={a['n']:4d} recovered {a['rec']:3d}/{a['lost']:3d}  ceiling {a['c0']/a['n']:.3f} -> {a['c1']/a['n']:.3f}")

if __name__ == "__main__":
    main()
