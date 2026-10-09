"""Loader tests on tiny synthetic fixtures (no network / downloaded data needed)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from bench.locomo import CATEGORY_NAMES, load_locomo, parse_session_datetime
from bench.longmemeval import load_longmemeval, parse_lme_datetime


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------
def _tiny_locomo_conv():
    return {
        "sample_id": "conv-test-1",
        "conversation": {
            "speaker_a": "Ann",
            "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ann", "dia_id": "D1:1", "text": "Hi Bob!"},
                {
                    "speaker": "Bob",
                    "dia_id": "D1:2",
                    "text": "Hey Ann, look at this",
                    "img_url": ["http://example.com/x.jpg"],
                    "blip_caption": "a cat sitting on a chair",
                },
            ],
            "session_2_date_time": "3:00 pm on 9 May, 2023",
            "session_2": [
                {"speaker": "Ann", "dia_id": "D2:1", "text": "How was your day?"},
            ],
        },
        "qa": [
            # category 4 = single-hop, valid
            {"question": "What did Bob show Ann?", "answer": "a cat photo", "evidence": ["D1:2"], "category": 4},
            # category 1 = multi-hop, valid, 2 evidence ids
            {"question": "Recap the chat", "answer": "greeting then photo", "evidence": ["D1:1", "D1:2"], "category": 1},
            # category 5 = adversarial -> must be excluded regardless of evidence
            {"question": "Adversarial Q", "answer": None, "evidence": ["D1:1"], "category": 5, "adversarial_answer": "false claim"},
            # empty evidence -> must be excluded regardless of category
            {"question": "No evidence Q", "answer": "?", "evidence": [], "category": 3},
        ],
    }


def test_locomo_loader_maps_fields_and_filters(tmp_path):
    data = [_tiny_locomo_conv()]
    p = tmp_path / "locomo_tiny.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    convs = load_locomo(p)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.conv_id == "conv-test-1"

    # 3 items total across 2 sessions
    assert len(conv.items) == 3
    by_id = {item.id: item for item in conv.items}

    item1 = by_id["D1:1"]
    assert item1.text == "Ann: Hi Bob!"
    assert item1.speaker == "Ann"
    assert item1.session_id == "session_1"

    # blip_caption appended as "[image: caption]"
    item2 = by_id["D1:2"]
    assert item2.text == "Bob: Hey Ann, look at this [image: a cat sitting on a chair]"
    assert item2.meta["has_image"] is True

    item3 = by_id["D2:1"]
    assert item3.session_id == "session_2"

    # event_time parsed from session_N_date_time, UTC
    expected_t1 = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp()
    assert item1.event_time == expected_t1
    expected_t2 = datetime(2023, 5, 9, 15, 0, tzinfo=timezone.utc).timestamp()
    assert item3.event_time == expected_t2

    # only the 2 valid questions (category 4 and 1) survive; 5 and empty-evidence are dropped
    assert len(conv.questions) == 2
    qids_by_text = {q.text: q for q in conv.questions}
    assert "Adversarial Q" not in qids_by_text
    assert "No evidence Q" not in qids_by_text

    q_single = qids_by_text["What did Bob show Ann?"]
    assert q_single.evidence_ids == ["D1:2"]
    assert q_single.category == "4"
    assert q_single.meta["category_name"] == "single-hop"
    assert q_single.query_time is None  # LoCoMo has no per-question date

    q_multi = qids_by_text["Recap the chat"]
    assert q_multi.evidence_ids == ["D1:1", "D1:2"]
    assert q_multi.category == "1"
    assert q_multi.meta["category_name"] == "multi-hop"


def test_locomo_category_names_mapping():
    assert CATEGORY_NAMES == {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    }


def test_parse_session_datetime_valid_and_invalid():
    ts = parse_session_datetime("10:04 am on 19 December, 2023")
    assert ts == datetime(2023, 12, 19, 10, 4, tzinfo=timezone.utc).timestamp()
    assert parse_session_datetime(None) is None
    assert parse_session_datetime("not a date") is None


def test_locomo_loader_text_without_image_has_no_bracket(tmp_path):
    data = [_tiny_locomo_conv()]
    p = tmp_path / "locomo_tiny2.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    conv = load_locomo(p)[0]
    plain_item = next(item for item in conv.items if item.id == "D1:1")
    assert "[image:" not in plain_item.text


# ---------------------------------------------------------------------------
# LongMemEval oracle
# ---------------------------------------------------------------------------
def _tiny_lme_entries():
    return [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What car did I mention?",
            "answer": "a Honda",
            "question_date": "2023/04/10 (Mon) 23:07",
            "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/04/09 (Sun) 09:00"],
            "haystack_session_ids": ["sess_a", "sess_b"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I got a Honda today.", "has_answer": True},
                    {"role": "assistant", "content": "Nice, congrats!", "has_answer": False},
                ],
                [
                    {"role": "user", "content": "Unrelated chat.", "has_answer": False},
                ],
            ],
            "answer_session_ids": ["sess_a", "sess_b"],
        },
        {
            # zero has_answer turns anywhere -> must be excluded entirely
            "question_id": "q2",
            "question_type": "multi-session",
            "question": "Synth question with no direct evidence turn",
            "answer": "synthesized",
            "question_date": "2023/04/11 (Tue) 08:00",
            "haystack_dates": ["2023/04/11 (Tue) 07:00"],
            "haystack_session_ids": ["sess_c"],
            "haystack_sessions": [
                [{"role": "user", "content": "Nothing relevant.", "has_answer": False}],
            ],
            "answer_session_ids": ["sess_c"],
        },
    ]


def test_longmemeval_loader_maps_fields_and_filters(tmp_path):
    p = tmp_path / "lme_tiny.json"
    p.write_text(json.dumps(_tiny_lme_entries()), encoding="utf-8")

    convs = load_longmemeval(p)
    # q2 is dropped (zero has_answer turns) -> only 1 conversation survives
    assert len(convs) == 1
    conv = convs[0]
    assert conv.conv_id == "q1"
    assert len(conv.items) == 3
    ids = [item.id for item in conv.items]
    assert ids == ["sess_a:0", "sess_a:1", "sess_b:0"]

    item0 = conv.items[0]
    assert item0.text == "I got a Honda today."
    assert item0.speaker == "user"
    assert item0.session_id == "sess_a"
    assert item0.meta["has_answer"] is True
    expected_t = datetime(2023, 4, 10, 17, 50, tzinfo=timezone.utc).timestamp()
    assert item0.event_time == expected_t

    assert len(conv.questions) == 1
    q = conv.questions[0]
    assert q.qid == "q1"
    assert q.evidence_ids == ["sess_a:0"]
    assert q.category == "single-session-user"
    assert q.query_time == datetime(2023, 4, 10, 23, 7, tzinfo=timezone.utc).timestamp()


def test_longmemeval_loader_respects_limit(tmp_path):
    entries = _tiny_lme_entries()
    # add a second valid (non-empty-evidence) entry so limit=1 has something to cut
    extra = json.loads(json.dumps(entries[0]))
    extra["question_id"] = "q3"
    p = tmp_path / "lme_tiny2.json"
    p.write_text(json.dumps(entries + [extra]), encoding="utf-8")

    convs_all = load_longmemeval(p)
    assert {c.conv_id for c in convs_all} == {"q1", "q3"}  # q2 always dropped

    convs_limited = load_longmemeval(p, limit=1)
    assert len(convs_limited) == 1


def test_parse_lme_datetime_valid_and_invalid():
    ts = parse_lme_datetime("2023/04/10 (Mon) 23:07")
    assert ts == datetime(2023, 4, 10, 23, 7, tzinfo=timezone.utc).timestamp()
    assert parse_lme_datetime(None) is None
    assert parse_lme_datetime("garbage") is None
