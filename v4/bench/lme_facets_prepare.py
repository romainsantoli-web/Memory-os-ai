"""Render the annotation inputs for the facet channel's SECOND benchmark
(LongMemEval oracle, the same stratified 150 questions every LME row of
RESULTS.md uses). Model-free, writes only under bench/data/enrich/.

    PYTHONPATH=src:. python -m bench.lme_facets_prepare

Outputs
- lme_facets_in/batch_XX.jsonl : {"tid", "session", "text"} -- USER turns only
  (assistant turns are generic advice: no personal facet to index; the
  channel is neutral on them by construction), text clipped to MAX_CHARS,
  grouped by session so the annotator sees a session's turns together.
- lme_questions/questions.jsonl : {"qid", "question"} -- question text ONLY
  (no answer, no category, no evidence), deterministically shuffled so
  neighbouring lines are unrelated users (the LoCoMo caveat "facets inferred
  from the neighbouring question" cannot recur).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from bench.run_bench import load_dataset

ENRICH = Path(__file__).parent / "data" / "enrich"
MAX_CHARS = 800
N_BATCHES = 8
LIMIT = 150


def main() -> None:
    convs = load_dataset("longmemeval_oracle", LIMIT, stratified=True)
    assert len(convs) == LIMIT, len(convs)

    seen: set[str] = set()
    sessions: list[list[dict]] = []
    for conv in convs:
        by_session: dict[str, list[dict]] = {}
        for item in conv.items:
            if item.speaker != "user" or item.id in seen or not item.text.strip():
                continue
            seen.add(item.id)
            by_session.setdefault(item.session_id, []).append(
                {"tid": item.id, "session": item.session_id, "text": item.text[:MAX_CHARS]}
            )
        sessions.extend(by_session.values())

    out_dir = ENRICH / "lme_facets_in"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_turns = sum(len(s) for s in sessions)
    target = n_turns / N_BATCHES
    batch: list[dict] = []
    batch_idx = 0
    sizes = []
    for sess in sessions:
        batch.extend(sess)
        if len(batch) >= target and batch_idx < N_BATCHES - 1:
            (out_dir / f"batch_{batch_idx:02d}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch)
            )
            sizes.append(len(batch))
            batch, batch_idx = [], batch_idx + 1
    if batch:
        (out_dir / f"batch_{batch_idx:02d}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch)
        )
        sizes.append(len(batch))

    q_dir = ENRICH / "lme_questions"
    q_dir.mkdir(parents=True, exist_ok=True)
    questions = [{"qid": c.questions[0].qid, "question": c.questions[0].text} for c in convs]
    random.Random(0).shuffle(questions)
    (q_dir / "questions.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions)
    )

    chars = sum(len(r["text"]) for s in sessions for r in s)
    print(f"user turns: {n_turns}  sessions: {len(sessions)}  chars: {chars}  batches: {sizes}")
    print(f"questions: {len(questions)}")


if __name__ == "__main__":
    main()
