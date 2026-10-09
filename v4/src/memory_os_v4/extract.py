"""Fact extraction: validating client-supplied facts, a heuristic extractor for
declarative sentences, and an optional LLM-backed extractor."""

from __future__ import annotations

from datetime import datetime

from .entities import extract_entities
from .llm import LLMClient
from .temporal import parse_absolute_dates
from .text import split_sentences, tokenize

VALID_FACT_KINDS = {"fact", "belief", "summary", "procedure"}


def _coerce_time(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def normalize_client_facts(facts: list[dict]) -> list[dict]:
    """Validate and normalize client-supplied fact dicts.

    Required: ``text``. Optional: ``kind``, ``entities``, ``event_time`` (unix
    or ISO string), ``valid_from``, ``valid_until``, ``importance``, ``tags``,
    ``confidence``, ``item_id``.
    """
    out = []
    for f in facts:
        if not isinstance(f, dict) or not f.get("text"):
            raise ValueError(f"fact must be a dict with a non-empty 'text' field: {f!r}")
        text = str(f["text"]).strip()
        if not text:
            raise ValueError("fact 'text' must not be blank")
        kind = f.get("kind", "fact")
        if kind not in VALID_FACT_KINDS:
            raise ValueError(f"invalid fact kind: {kind!r}")

        event_time = _coerce_time(f.get("event_time"))
        tags = list(f.get("tags") or [])
        if event_time is None:
            # A7(a) (adversarial review, 2026-09-12): `temporal.parse_dates`
            # was implemented but never wired into fact creation -- a fact
            # whose text carries an explicit date ("Deployed on 2026-09-12")
            # but no `event_time` field silently defaulted to "now" instead
            # of the date the fact is actually about. Only ABSOLUTE dates
            # count here (`parse_absolute_dates`, not `parse_dates`): a
            # relative expression like "yesterday" has no well-defined
            # anchor at fact-creation time. Uses the FIRST date found; tags
            # the fact so callers/tests can tell the event_time was inferred.
            dates = parse_absolute_dates(text)
            if dates:
                event_time = dates[0]
                if "date_from_text" not in tags:
                    tags.append("date_from_text")

        norm = {
            "text": text,
            "kind": kind,
            "entities": f.get("entities") or extract_entities(text),
            "event_time": event_time,
            "valid_from": _coerce_time(f.get("valid_from")),
            "valid_until": _coerce_time(f.get("valid_until")),
            "importance": float(f.get("importance", 0.5)),
            "tags": tags,
            "confidence": float(f.get("confidence", 1.0)),
        }
        identity = f.get("identity")
        if identity is not None:
            if not isinstance(identity, dict) or set(identity) != {"subject_id", "predicate", "cardinality"}:
                raise ValueError("identity requires subject_id, predicate and cardinality")
            if not all(isinstance(identity[k], str) and identity[k].strip() for k in ("subject_id", "predicate")):
                raise ValueError("identity subject_id and predicate must be non-empty strings")
            if identity["cardinality"] not in {"single", "multi"}:
                raise ValueError("identity cardinality must be single or multi")
            norm["identity"] = {k: v.strip() for k, v in identity.items()}
        replaces = f.get("replaces") or []
        if not isinstance(replaces, list) or not all(isinstance(x, str) and x for x in replaces):
            raise ValueError("replaces must be a list of item ids")
        norm["replaces"] = list(dict.fromkeys(replaces))
        if (identity is not None or replaces) and kind != "fact":
            raise ValueError("identity and replaces are supported only for facts")
        if f.get("item_id"):
            norm["item_id"] = f["item_id"]
        out.append(norm)
    return out


def heuristic_facts(text: str, max_facts: int = 8) -> list[dict]:
    """Pull declarative sentences (>=6 tokens, no '?', contains an entity or a
    number) as candidate atomic facts, de-duplicated."""
    facts: list[dict] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        if "?" in sentence:
            continue
        toks = tokenize(sentence)
        if len(toks) < 6:
            continue
        ents = extract_entities(sentence)
        has_number = any(any(ch.isdigit() for ch in t) for t in toks)
        if not ents and not has_number:
            continue
        key = sentence.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append({"text": sentence.strip(), "kind": "fact", "entities": ents})
        if len(facts) >= max_facts:
            break
    return facts


def llm_facts(
    text: str,
    existing_snippets: list[str] | None = None,
    llm_client: LLMClient | None = None,
) -> list[dict]:
    """Extract facts via an LLM backend, if configured and reachable."""
    client = llm_client or LLMClient()
    if not client.available():
        return []
    raw = client.extract_facts(text, existing_snippets)
    try:
        return normalize_client_facts(raw)
    except ValueError:
        return []
