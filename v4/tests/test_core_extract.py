"""Client fact validation, heuristic fact extraction, and the LLM client stub."""

import pytest

from memory_os_v4.extract import heuristic_facts, llm_facts, normalize_client_facts
from memory_os_v4.llm import LLMClient


def test_normalize_client_facts_requires_text():
    with pytest.raises(ValueError):
        normalize_client_facts([{"kind": "fact"}])


def test_normalize_client_facts_rejects_invalid_kind():
    with pytest.raises(ValueError):
        normalize_client_facts([{"text": "x", "kind": "nonsense"}])


def test_normalize_client_facts_fills_defaults():
    out = normalize_client_facts([{"text": "Romain lives in Paris."}])
    assert out[0]["kind"] == "fact"
    assert "paris" in out[0]["entities"]
    assert out[0]["importance"] == 0.5
    assert out[0]["confidence"] == 1.0


def test_normalize_client_facts_coerces_iso_event_time():
    out = normalize_client_facts([{"text": "x entity Paris", "event_time": "2026-09-12T00:00:00+00:00"}])
    assert out[0]["event_time"] is not None
    assert isinstance(out[0]["event_time"], float)


def test_normalize_client_facts_accepts_unix_event_time():
    out = normalize_client_facts([{"text": "x entity Paris", "event_time": 1700000000.0}])
    assert out[0]["event_time"] == 1700000000.0


# ---------------------------------------------------------------------------
# A7(a): a fact with no event_time but an absolute date in its text gets
# that date as event_time, tagged "date_from_text" (adversarial review,
# 2026-09-12: temporal.parse_dates existed but was never wired in).
# ---------------------------------------------------------------------------
def test_normalize_client_facts_infers_event_time_from_absolute_date_in_text():
    from datetime import datetime, timezone

    out = normalize_client_facts([{"text": "Deployed the new pipeline on 2026-09-12."}])
    assert out[0]["event_time"] == datetime(2026, 9, 12, tzinfo=timezone.utc).timestamp()
    assert "date_from_text" in out[0]["tags"]


def test_normalize_client_facts_does_not_infer_from_relative_date_text():
    """Relative expressions ('yesterday') have no well-defined anchor at
    fact-creation time -- only ABSOLUTE dates are inferred."""
    out = normalize_client_facts([{"text": "We shipped the release yesterday."}])
    assert out[0]["event_time"] is None
    assert "date_from_text" not in out[0]["tags"]


def test_normalize_client_facts_explicit_event_time_wins_over_text_date():
    out = normalize_client_facts(
        [{"text": "Deployed on 2026-09-12.", "event_time": 1700000000.0}]
    )
    assert out[0]["event_time"] == 1700000000.0
    assert "date_from_text" not in out[0]["tags"]


def test_normalize_client_facts_no_date_anywhere_leaves_event_time_none():
    out = normalize_client_facts([{"text": "Romain lives in Paris."}])
    assert out[0]["event_time"] is None
    assert "date_from_text" not in out[0]["tags"]


def test_heuristic_facts_picks_declarative_sentences_with_entities_or_numbers():
    text = (
        "How are you doing today? "
        "Romain deployed the new xgb_v14 model to production yesterday. "
        "Latency dropped significantly by 12ms after the recent change was deployed."
    )
    facts = heuristic_facts(text)
    texts = [f["text"] for f in facts]
    assert any("xgb_v14" in t for t in texts)
    assert any("12ms" in t for t in texts)
    assert not any("?" in t for t in texts)


def test_heuristic_facts_respects_max_facts():
    text = " ".join(f"Romain shipped feature number {i} to project Falcon successfully." for i in range(20))
    facts = heuristic_facts(text, max_facts=3)
    assert len(facts) <= 3


def test_heuristic_facts_dedups_identical_sentences():
    sentence = "Romain successfully shipped the new feature called Falcon today."
    facts = heuristic_facts(f"{sentence} {sentence}")
    assert len(facts) == 1


def test_heuristic_facts_skips_short_sentences():
    facts = heuristic_facts("Yes. No. Ok then.")
    assert facts == []


def test_llm_client_unavailable_by_default(monkeypatch):
    monkeypatch.delenv("MEMORY_V4_LLM_URL", raising=False)
    client = LLMClient()
    assert client.available() is False
    assert client.extract_facts("some text") == []


def test_llm_facts_returns_empty_when_unavailable(monkeypatch):
    monkeypatch.delenv("MEMORY_V4_LLM_URL", raising=False)
    assert llm_facts("some text") == []
