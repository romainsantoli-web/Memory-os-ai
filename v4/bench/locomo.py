"""Loader for the LoCoMo10 benchmark (snap-research/locomo).

JSON structure (inspected empirically on bench/data/locomo10.json, 10 entries):

    conv = data[i]                              # one of 10 conversations
    conv.keys() -> qa, conversation, event_summary, observation,
                   session_summary, sample_id
    conv["conversation"].keys() -> speaker_a, speaker_b,
        session_1_date_time, session_1, session_2_date_time, session_2, ...
    conv["conversation"]["session_N"] -> list[turn]
        turn.keys() -> speaker, dia_id, text, [img_url, blip_caption, query]
        dia_id looks like "D<N>:<k>" (always matches its own session number).
    conv["conversation"]["session_N_date_time"] -> "1:56 pm on 8 May, 2023"
    conv["qa"] -> list of {question, answer, evidence: [dia_id, ...], category,
                            adversarial_answer (category 5 only)}

Category mapping — VERIFIED EMPIRICALLY (not just asserted from the prompt) by
reading examples and computing evidence-count statistics over the full corpus
(2825 questions total):

    cat  n     avg_len(evidence)  frac(len>=2)   examples read
    1   282    3.13               97.9%          "What did Caroline research?" (multi-fact) -> MULTI-HOP
    2   321    1.17               12.5%          "When did Caroline go to the LGBTQ support group?" -> TEMPORAL
    3    96    2.08               49.0%          "Would Caroline still want to pursue counseling...if she
                                                   hadn't received support growing up?" (inference beyond
                                                   any single stated fact) -> OPEN-DOMAIN (commonsense/inference)
    4   841    1.07               5.5%           "What did the charity race raise awareness for?" (one direct
                                                   fact) -> SINGLE-HOP
    5   446    1.03               3.1%           answer is always null, `adversarial_answer` is the plausible-
                                                   but-false claim, evidence is the fact the false claim
                                                   contradicts -> ADVERSARIAL (no ground-truth answer to
                                                   retrieve *for*, so excluded from the bench)

    This matches exactly the mapping given in the task brief: 1=multi-hop,
    2=temporal, 3=open-domain, 4=single-hop, 5=adversarial. Category 1's very
    high multi-evidence fraction (97.9%) and category 4's very low one (5.5%)
    make the multi-hop/single-hop split unambiguous; category 2's questions
    are uniformly "When did ..." (temporal); category 3's are counterfactual/
    inferential "Would X..."/"What fields would X..." questions not directly
    answerable from one quoted sentence (open-domain reasoning).

    4 questions (all in category 3) have an empty `evidence` list in the raw
    data; these are dropped by `load_locomo` regardless of category, per the
    bench brief ("exclude ... any question with empty evidence").

    9 / 2815 (0.3%) evidence dia_ids reference turns that do not exist in
    their own conversation (a symptom of the ~6% LoCoMo answer-key error rate
    documented independently, see docs/RESEARCH_NOTES.md) — left as-is (they
    simply can never be recalled, which is the correct, honest behaviour).

`session_N_date_time` strings all follow "H:MM am/pm on D Month, YYYY"
(checked: 287 distinct instances, zero parse failures). Parsed as UTC-naive
(`%I:%M %p on %d %B, %Y`, tagged tzinfo=UTC) so timestamps are reproducible
regardless of the machine's local timezone. LoCoMo carries no per-question
timestamp, so `BenchQuestion.query_time` is always None here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bench.protocol import BenchConversation, BenchItem, BenchQuestion

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "locomo10.json"

_DATE_FMT = "%I:%M %p on %d %B, %Y"

# Verified mapping (see module docstring). Kept explicit rather than
# hard-coding "5" everywhere so the exclusion rule reads clearly at call sites.
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
EXCLUDED_CATEGORIES = {5}


def parse_session_datetime(s: Optional[str]) -> Optional[float]:
    """Parse a LoCoMo `session_N_date_time` string to a UTC unix timestamp.

    Returns None (rather than raising) on anything that doesn't match the
    one observed format, so a single malformed date can't crash a whole run.
    """
    if not s:
        return None
    try:
        dt = datetime.strptime(s.strip(), _DATE_FMT)
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _turn_text(turn: dict[str, Any]) -> str:
    speaker = turn.get("speaker", "")
    text = turn.get("text", "") or ""
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [image: {caption}]".strip() if text else f"[image: {caption}]"
    return f"{speaker}: {text}"


def _load_conversation(conv: dict[str, Any], conv_idx: int) -> BenchConversation:
    convo = conv["conversation"]
    conv_id = str(conv.get("sample_id", conv_idx))

    items: list[BenchItem] = []
    session_num = 1
    while f"session_{session_num}" in convo:
        session_key = f"session_{session_num}"
        date_key = f"{session_key}_date_time"
        event_time = parse_session_datetime(convo.get(date_key))
        for turn in convo[session_key]:
            items.append(
                BenchItem(
                    id=turn["dia_id"],
                    text=_turn_text(turn),
                    speaker=turn.get("speaker", ""),
                    event_time=event_time,
                    session_id=session_key,
                    meta={
                        "has_image": "blip_caption" in turn,
                        "conv_id": conv_id,
                    },
                )
            )
        session_num += 1

    questions: list[BenchQuestion] = []
    for qi, qa in enumerate(conv.get("qa", [])):
        category = qa.get("category")
        evidence = qa.get("evidence") or []
        if category in EXCLUDED_CATEGORIES:
            continue
        if not evidence:
            continue
        questions.append(
            BenchQuestion(
                qid=f"{conv_id}:q{qi}",
                text=qa["question"],
                evidence_ids=list(evidence),
                category=str(category),
                query_time=None,
                meta={"answer": qa.get("answer"), "category_name": CATEGORY_NAMES.get(category, str(category))},
            )
        )

    return BenchConversation(conv_id=conv_id, items=items, questions=questions)


def load_locomo(path: str | Path = DEFAULT_PATH) -> list[BenchConversation]:
    """Load every conversation in locomo10.json as a BenchConversation."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [_load_conversation(conv, i) for i, conv in enumerate(data)]
