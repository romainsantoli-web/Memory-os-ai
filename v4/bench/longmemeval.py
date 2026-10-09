"""Loader for the LongMemEval **oracle** variant (xiaowu0162/longmemeval on HF).

JSON structure (inspected on bench/data/longmemeval_oracle.json, 500 entries,
top-level a plain list — no wrapping object):

    entry.keys() -> question_id, question_type, question, answer, question_date,
                    haystack_dates, haystack_session_ids, haystack_sessions,
                    answer_session_ids
    haystack_dates / haystack_session_ids / haystack_sessions are three
    parallel lists (verified: always equal length, zero mismatches).
    haystack_sessions[i] -> list[turn], turn = {role, content, has_answer}
        role in {"user", "assistant"}; has_answer is a bool.
    question_date / haystack_dates format: "2023/04/10 (Mon) 23:07"
        (zero parse failures over the whole file with "%Y/%m/%d (%a) %H:%M").

Oracle-variant note (verified empirically): for all 500 questions,
`haystack_session_ids == answer_session_ids` exactly — i.e. unlike the -s/-m
variants there are NO distractor sessions in the oracle haystack, only the
sessions known to be relevant to the answer. Turn-level evidence is still
sparse though: only 896 / 10960 turns (8.2%) have `has_answer == True`, and
21 / 500 questions have *zero* has_answer turns in their haystack (mostly
"multi-session" / "knowledge-update" questions whose answer is a synthesis
rather than a single quoted turn). Those 21 are dropped by `load_longmemeval`
(empty evidence, same rule as LoCoMo) — evaluating turn-level recall against
an empty gold set is meaningless.

Each question becomes its OWN `BenchConversation` (its haystack is private to
it — session ids are reused across sibling questions in the raw data, e.g.
"89941a93" vs "89941a94", so item ids are only made globally-unique by being
scoped to one BenchConversation per question, never merged across questions).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from bench.protocol import BenchConversation, BenchItem, BenchQuestion

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "longmemeval_oracle.json"

_DATE_FMT = "%Y/%m/%d (%a) %H:%M"


def parse_lme_datetime(s: Optional[str]) -> Optional[float]:
    """Parse a LongMemEval date string ("2023/04/10 (Mon) 23:07") to UTC unix time."""
    if not s:
        return None
    try:
        dt = datetime.strptime(s.strip(), _DATE_FMT)
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _build_conversation(entry: dict[str, Any]) -> BenchConversation:
    qid = entry["question_id"]
    items: list[BenchItem] = []
    evidence_ids: list[str] = []

    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    session_dates = entry["haystack_dates"]

    for session_id, session_date, turns in zip(session_ids, session_dates, sessions):
        event_time = parse_lme_datetime(session_date)
        for turn_idx, turn in enumerate(turns):
            item_id = f"{session_id}:{turn_idx}"
            items.append(
                BenchItem(
                    id=item_id,
                    text=turn.get("content", "") or "",
                    speaker=turn.get("role", ""),
                    event_time=event_time,
                    session_id=session_id,
                    meta={"has_answer": bool(turn.get("has_answer", False))},
                )
            )
            if turn.get("has_answer"):
                evidence_ids.append(item_id)

    question = BenchQuestion(
        qid=qid,
        text=entry["question"],
        evidence_ids=evidence_ids,
        category=entry.get("question_type", "unknown"),
        query_time=parse_lme_datetime(entry.get("question_date")),
        meta={"answer": entry.get("answer")},
    )
    return BenchConversation(conv_id=qid, items=items, questions=[question])


def iter_longmemeval(path: str | Path = DEFAULT_PATH, limit: Optional[int] = None) -> Iterator[BenchConversation]:
    """Yield one BenchConversation per question, lazily.

    The oracle file is small enough (~15 MB / 500 entries) that `json.load`
    parses it in one shot, but conversations (with their embeddings/index
    state built downstream in run_bench.py) are handed out and can be
    discarded one at a time instead of all being held in memory together —
    that's the "don't hold all haystacks in memory at once" requirement:
    it is the per-question *retriever* state that must stay bounded, and
    iterating here (rather than returning one big list) lets the caller
    reset/drop each retriever before moving to the next question.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    for entry in data:
        conv = _build_conversation(entry)
        if not conv.questions[0].evidence_ids:
            continue  # no has_answer turns at all: nothing recoverable, drop
        yield conv
        n += 1
        if limit is not None and n >= limit:
            return


def load_longmemeval(path: str | Path = DEFAULT_PATH, limit: Optional[int] = None) -> list[BenchConversation]:
    """Eager convenience wrapper around `iter_longmemeval` (used by tests)."""
    return list(iter_longmemeval(path, limit=limit))
