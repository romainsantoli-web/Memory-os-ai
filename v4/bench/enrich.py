"""Loader + coverage reporting for offline LLM enrichment of LoCoMo turns.

Enrichment files are produced *concurrently*, by other worker sessions, at
``bench/data/enrich/locomo_facts/<conv_id>.jsonl`` — one JSON object per
line, one line per turn::

    {"turn_id": "D1:2", "facts": ["..."], "keywords": ["..."], "entities": ["..."]}

Because those files are written while this module (and the bench run that
uses it) may be executing, every function here MUST be tolerant of:

- the file not existing yet at all (a conversation whose worker hasn't
  started/finished) -> treated as "no enrichment for this conversation",
  never an error;
- a line that is not valid JSON, or valid JSON that isn't a well-shaped
  record (missing/blank ``turn_id``, non-list ``facts``/``keywords``/
  ``entities``) -> skipped and counted, never raised;
- a truncated last line (a worker's write caught mid-flush) -> same as
  malformed: skipped and counted, the rest of the file is still read;
- the same ``turn_id`` appearing more than once (a worker overwriting its
  own partial output by re-emitting a turn later in the file) -> last
  occurrence wins.

Nothing in this module ever mutates an enrichment file — it is read-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class EnrichRecord:
    """One turn's offline LLM enrichment."""

    facts: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


def _clean_str_list(value: Any) -> list[str] | None:
    """Return a list[str] (blanks/non-strings dropped) or None if `value`
    isn't list-shaped at all (that's a malformed record, not just an empty
    list -- an empty list is perfectly valid, e.g. a turn with no facts)."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]


def _parse_line(line: str) -> tuple[str, EnrichRecord] | None:
    """Parse one enrichment JSONL line. Returns None (never raises) if the
    line is blank, not valid JSON, not an object, or missing/blank
    `turn_id`, or if `facts`/`keywords`/`entities` are present but not
    list-shaped."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    turn_id = obj.get("turn_id")
    if not isinstance(turn_id, str) or not turn_id.strip():
        return None
    facts = _clean_str_list(obj.get("facts"))
    keywords = _clean_str_list(obj.get("keywords"))
    entities = _clean_str_list(obj.get("entities"))
    if facts is None or keywords is None or entities is None:
        return None
    return turn_id, EnrichRecord(facts=facts, keywords=keywords, entities=entities)


def _load_enrichment_with_stats(
    enrich_dir: str | Path, conv_id: str
) -> tuple[dict[str, EnrichRecord], dict[str, int]]:
    """Shared implementation behind `load_enrichment` and `coverage_report`.

    Returns `(records, stats)` where `stats` carries `n_lines` (non-blank
    lines seen), `n_malformed` (of those, how many failed to parse), and
    `file_found` (0/1, so callers can tell "no file yet" apart from "file
    present but every line malformed").
    """
    path = Path(enrich_dir) / f"{conv_id}.jsonl"
    records: dict[str, EnrichRecord] = {}
    n_lines = 0
    n_malformed = 0
    if not path.exists():
        return records, {"n_lines": 0, "n_malformed": 0, "file_found": 0}
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        # File vanished/unreadable between exists() and read (a worker
        # mid-rewrite) -- treat exactly like "no file yet" rather than raising.
        return records, {"n_lines": 0, "n_malformed": 0, "file_found": 0}
    for raw_line in raw_lines:
        if not raw_line.strip():
            continue
        n_lines += 1
        parsed = _parse_line(raw_line)
        if parsed is None:
            n_malformed += 1
            continue
        turn_id, record = parsed
        records[turn_id] = record  # duplicates: last one wins
    return records, {"n_lines": n_lines, "n_malformed": n_malformed, "file_found": 1}


def load_enrichment(enrich_dir: str | Path, conv_id: str) -> dict[str, EnrichRecord]:
    """Load `<enrich_dir>/<conv_id>.jsonl` into `{turn_id: EnrichRecord}`.

    Never raises: a missing file returns `{}`; malformed lines are silently
    skipped (use `coverage_report` for visibility into how many); a
    duplicate `turn_id` keeps its last occurrence.
    """
    records, _stats = _load_enrichment_with_stats(enrich_dir, conv_id)
    return records


# ---------------------------------------------------------------------------
# P9 step A: offline LLM query planning (bench/data/enrich/locomo_queries/).
# ---------------------------------------------------------------------------
@dataclass
class QueryPlan:
    """One question's offline LLM-generated retrieval plan (P9 §A, PLAN.md).

    Produced concurrently by other worker sessions at
    ``bench/data/enrich/locomo_queries/<conv_id>.jsonl``, one JSON object per
    line, one line per question::

        {"qid": "conv-30:q0", "subqueries": ["..."], "hypotheses": ["..."]}

    ``qid`` matches ``BenchQuestion.qid`` (``f"{conv_id}:q{qi}"``, see
    ``bench/locomo.py``), which is how ``bench.retrievers.V4PlannedRetriever``
    looks up the plan for a given question at search time.
    """

    subqueries: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)


def _parse_query_line(line: str) -> tuple[str, QueryPlan] | None:
    """Parse one query-plan JSONL line. Same tolerance contract as
    `_parse_line` above: returns None (never raises) for a blank line,
    invalid JSON, a non-object, a missing/blank `qid`, or a `subqueries`/
    `hypotheses` value that is present but not list-shaped."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    qid = obj.get("qid")
    if not isinstance(qid, str) or not qid.strip():
        return None
    subqueries = _clean_str_list(obj.get("subqueries"))
    hypotheses = _clean_str_list(obj.get("hypotheses"))
    if subqueries is None or hypotheses is None:
        return None
    return qid, QueryPlan(subqueries=subqueries, hypotheses=hypotheses)


def load_query_plans(queries_dir: str | Path, conv_id: str) -> dict[str, QueryPlan]:
    """Load `<queries_dir>/<conv_id>.jsonl` into `{qid: QueryPlan}`.

    Never raises: a missing file (a conversation whose planning worker
    hasn't started/finished, or a run that opted out of `--queries-dir`)
    returns `{}` -- callers (see `bench.retrievers.V4PlannedRetriever`) fall
    back to single-query behaviour per missing `qid`, never an error.
    Malformed lines are silently skipped; a duplicate `qid` keeps its last
    occurrence, matching `load_enrichment`'s contract exactly (see module
    docstring for why: files are written concurrently by other workers).
    """
    path = Path(queries_dir) / f"{conv_id}.jsonl"
    plans: dict[str, QueryPlan] = {}
    if not path.exists():
        return plans
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return plans
    for raw_line in raw_lines:
        if not raw_line.strip():
            continue
        parsed = _parse_query_line(raw_line)
        if parsed is None:
            continue
        qid, plan = parsed
        plans[qid] = plan  # duplicates: last one wins
    return plans


# ---------------------------------------------------------------------------
# Idea 8 (docs/IDEAS.md, 2026-09-19): facet channel + implications channel.
#
# Two data sources, both produced offline by other worker sessions, COMPLETE
# and read-only (never mutated here, per the task brief):
#
# - ``bench/data/enrich/locomo_facets/<conv_id>.jsonl`` -- one line per
#   extracted fact: ``{"fid": "<turn_id>#<i>", "facets": [{"subject",
#   "predicate", "object"}], "implications": [str]}``. The turn id is the
#   part of ``fid`` before ``"#"`` (`fid_to_turn_id`) -- unique only WITHIN a
#   conversation, matching every other per-conversation id in this bench.
#   NOTE: the real directory also holds `*.part*.jsonl`, `*.impl.jsonl` and
#   `*.facets_only_v1.jsonl` scratch files from earlier drafts (see
#   docs/IDEAS.md) -- `load_fact_facets` only ever reads the exact
#   ``<conv_id>.jsonl`` path, so those are never touched.
# - ``bench/data/enrich/locomo_qfacets/<conv_id>.jsonl`` -- one line per
#   question: ``{"qid", "facets": [{"subject", "predicate"}]}`` (no
#   ``object`` -- a question facet is a query pattern, not a value).
#
# Same tolerance contract as the rest of this module: a missing file is "no
# facet data for this conversation", never an error; malformed lines/entries
# are skipped; a duplicate fid/qid keeps its last occurrence.
# ---------------------------------------------------------------------------
@dataclass
class FactFacetRecord:
    """One extracted fact's facets + implications (`locomo_facets/<conv_id>.jsonl`)."""

    facets: list[dict[str, str]] = field(default_factory=list)
    implications: list[str] = field(default_factory=list)


def fid_to_turn_id(fid: str) -> str:
    """``"D5:1#0"`` -> ``"D5:1"`` -- the part of a fact id before ``"#"``
    (per the task brief: turn ids are unique only WITHIN a conversation,
    never across conversations -- callers must already be working with one
    conversation's own `fid`s, e.g. from a single `load_fact_facets` call)."""
    return fid.split("#", 1)[0]


def _clean_facet_triples(value: Any) -> list[dict[str, str]] | None:
    """Clean a ``facets`` list of ``{"subject", "predicate", "object"}``
    dicts (extracted-fact side): non-dict entries, and entries with a
    missing/blank ``subject`` or ``predicate``, are dropped (not fatal --
    `value` itself must still be list-shaped, same convention as
    `_clean_str_list`). ``object`` is optional and defaults to ``""``."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    out: list[dict[str, str]] = []
    for f in value:
        if not isinstance(f, dict):
            continue
        subject, predicate = f.get("subject"), f.get("predicate")
        if not isinstance(subject, str) or not subject.strip():
            continue
        if not isinstance(predicate, str) or not predicate.strip():
            continue
        obj = f.get("object")
        out.append(
            {
                "subject": subject.strip(),
                "predicate": predicate.strip(),
                "object": obj.strip() if isinstance(obj, str) else "",
            }
        )
    return out


def _parse_fact_facet_line(line: str) -> tuple[str, FactFacetRecord] | None:
    """Parse one `locomo_facets/<conv_id>.jsonl` line. Same tolerance
    contract as `_parse_line`: None (never raises) for a blank line, invalid
    JSON, a non-object, a missing/blank ``fid`` (or one with no ``"#"`` --
    not a well-shaped fact id), or a ``facets``/``implications`` value that
    is present but not list-shaped."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    fid = obj.get("fid")
    if not isinstance(fid, str) or not fid.strip() or "#" not in fid:
        return None
    facets = _clean_facet_triples(obj.get("facets"))
    implications = _clean_str_list(obj.get("implications"))
    if facets is None or implications is None:
        return None
    return fid, FactFacetRecord(facets=facets, implications=implications)


def load_fact_facets(facets_dir: str | Path, conv_id: str) -> dict[str, FactFacetRecord]:
    """Load `<facets_dir>/<conv_id>.jsonl` into `{fid: FactFacetRecord}`.

    Never raises: a missing file returns `{}` (a conversation with no facet
    data behaves exactly like today -- see `bench.retrievers.
    V4PlannedRetriever`'s facet channel); malformed lines are silently
    skipped; a duplicate `fid` keeps its last occurrence.
    """
    path = Path(facets_dir) / f"{conv_id}.jsonl"
    records: dict[str, FactFacetRecord] = {}
    if not path.exists():
        return records
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for raw_line in raw_lines:
        parsed = _parse_fact_facet_line(raw_line)
        if parsed is None:
            continue
        fid, record = parsed
        records[fid] = record  # duplicates: last one wins
    return records


def _clean_qfacet_pairs(value: Any) -> list[tuple[str, str]] | None:
    """Clean a question's ``facets`` list of ``{"subject", "predicate"}``
    dicts (no ``object`` -- a question facet is a query pattern). Same
    drop-bad-entries-not-fatal convention as `_clean_facet_triples`."""
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    out: list[tuple[str, str]] = []
    for f in value:
        if not isinstance(f, dict):
            continue
        subject, predicate = f.get("subject"), f.get("predicate")
        if not isinstance(subject, str) or not subject.strip():
            continue
        if not isinstance(predicate, str) or not predicate.strip():
            continue
        out.append((subject.strip(), predicate.strip()))
    return out


def _parse_qfacet_line(line: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Parse one `locomo_qfacets/<conv_id>.jsonl` line. Same tolerance
    contract as `_parse_query_line`: None (never raises) for a blank line,
    invalid JSON, a non-object, a missing/blank ``qid``, or a ``facets``
    value that is present but not list-shaped."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    qid = obj.get("qid")
    if not isinstance(qid, str) or not qid.strip():
        return None
    facets = _clean_qfacet_pairs(obj.get("facets"))
    if facets is None:
        return None
    return qid, facets


def load_qfacets(qfacets_dir: str | Path, conv_id: str) -> dict[str, list[tuple[str, str]]]:
    """Load `<qfacets_dir>/<conv_id>.jsonl` into `{qid: [(subject, predicate), ...]}`.

    Never raises: a missing file (no qfacets data, or a run that opted out
    of `--facet-channel`) returns `{}` -- a question with no entry here (or
    an empty `facets` list) means "no facets for this question", handled
    identically by `bench.retrievers.V4PlannedRetriever.search` (empty pool,
    no-op, exactly today's behaviour). Malformed lines are skipped; a
    duplicate `qid` keeps its last occurrence.
    """
    path = Path(qfacets_dir) / f"{conv_id}.jsonl"
    plans: dict[str, list[tuple[str, str]]] = {}
    if not path.exists():
        return plans
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return plans
    for raw_line in raw_lines:
        parsed = _parse_qfacet_line(raw_line)
        if parsed is None:
            continue
        qid, facets = parsed
        plans[qid] = facets  # duplicates: last one wins
    return plans


# ---------------------------------------------------------------------------
# P9 step D: LLM listwise reranking, offline (PLAN.md).
#
# Two artifacts on top of everything above:
#
# 1. **Candidate dumps** (`--dump-candidates DIR`, `bench/run_bench.py`):
#    for `--retriever v4_planned`, one line per question at
#    ``DIR/<conv_id>.jsonl`` recording the retriever's FINAL top-20 turn
#    ranking (after the bench's own cross-encoder rerank) in a compact,
#    LLM-readable form -- built by `build_candidate_dump_line` below and
#    read back by `load_candidate_dump`. Never carries evidence ids,
#    answers, or categories (an LLM listwise-reranking worker must not see
#    the gold labels it is meant to help predict).
# 2. **LLM rankings** (produced offline, later, by other worker sessions, at
#    ``<llmrank_dir>/<conv_id>.jsonl``): one line per question,
#    ``{"qid", "ranked": ["c07", "c02", ...]}`` -- labels (not turn ids),
#    best first, possibly fewer than the dump's candidate count. Read back
#    by `load_llm_rankings` and applied by
#    `bench.retrievers.V4LLMRerankedRetriever`/`apply_llm_ranking`.
#
# Same tolerance contract as the rest of this module throughout: a missing
# file is "nothing to apply yet", never an error; malformed lines are
# skipped; a duplicate qid keeps its last occurrence.
# ---------------------------------------------------------------------------
import os as _os
MAX_DUMP_CANDIDATES = int(_os.environ.get("BENCH_DUMP_MAX", "20"))  # P9: 20 by default, 50 for the wide dump
DUMP_TEXT_MAX_CHARS = 320
DUMP_FACT_MAX_CHARS = 160
DUMP_MAX_FACTS_PER_TURN = 2


def _label_for_rank(rank: int) -> str:
    """0-based rank -> ``"c01"``..``"c20"`` (etc.) candidate label."""
    return f"c{rank + 1:02d}"


@dataclass
class CandidateDumpRecord:
    """One question's `--dump-candidates` line, as read back by
    `load_candidate_dump` for `bench.retrievers.V4LLMRerankedRetriever` (and
    `bench.run_bench`'s `--from-dump` evaluation path).

    ``turn_ids``: the dumped top-N turn ids, best first (rank order).
    ``label_map``: the dump's own compact ``{"c01": turn_id, ...}`` mapping
    -- callers MUST use this (not recompute label positions themselves) to
    translate an LLM ranking's labels back to turn ids, per the class
    docstring of `bench.retrievers.V4LLMRerankedRetriever`.
    """

    qid: str
    question: str = ""
    turn_ids: list[str] = field(default_factory=list)
    label_map: dict[str, str] = field(default_factory=dict)


def build_candidate_dump_line(
    qid: str,
    question: str,
    ranked_turn_ids: Sequence[str],
    text_by_id: dict[str, str],
    date_by_id: dict[str, Optional[str]],
    speaker_by_id: dict[str, str],
    facts_by_id: dict[str, Sequence[str]],
    max_candidates: int = MAX_DUMP_CANDIDATES,
) -> dict[str, Any]:
    """Build one JSON-serializable `--dump-candidates` line for a question's
    FINAL top-`max_candidates` ranking (`bench/run_bench.py`).

    Truncates each candidate's ``text`` to `DUMP_TEXT_MAX_CHARS` chars and up
    to `DUMP_MAX_FACTS_PER_TURN` of its ``facts_by_id`` entries to
    `DUMP_FACT_MAX_CHARS` chars each. Returns exactly
    ``{"qid", "question", "candidates": [...], "label_map": {...}}`` --
    every candidate dict carries exactly ``{"label", "turn_id", "date",
    "speaker", "text", "facts"}``, nothing else (in particular: never an
    evidence id, an answer, or a category -- callers must not pass them in,
    and this function has no parameter that would accept them).
    """
    top_ids = list(ranked_turn_ids)[:max_candidates]
    candidates: list[dict[str, Any]] = []
    label_map: dict[str, str] = {}
    for rank, turn_id in enumerate(top_ids):
        label = _label_for_rank(rank)
        label_map[label] = turn_id
        facts = [
            fact[:DUMP_FACT_MAX_CHARS] for fact in list(facts_by_id.get(turn_id, ()))[:DUMP_MAX_FACTS_PER_TURN]
        ]
        candidates.append(
            {
                "label": label,
                "turn_id": turn_id,
                "date": date_by_id.get(turn_id),
                "speaker": speaker_by_id.get(turn_id, ""),
                "text": text_by_id.get(turn_id, "")[:DUMP_TEXT_MAX_CHARS],
                "facts": facts,
            }
        )
    return {
        "qid": qid,
        "question": question,
        "candidates": candidates,
        "label_map": label_map,
    }


def _parse_candidate_dump_line(line: str) -> tuple[str, CandidateDumpRecord] | None:
    """Parse one `--dump-candidates` JSONL line. Same tolerance contract as
    `_parse_line`: returns None (never raises) for a blank line, invalid
    JSON, a non-object, a missing/blank `qid`, or a non-list `candidates`.
    A malformed individual candidate entry (not an object, or missing
    `label`/`turn_id`) is skipped rather than failing the whole line. The
    top-level ``label_map`` field is preferred when present and well-shaped
    (a dict of str->str); otherwise it is rebuilt from `candidates` --
    tolerates a dump written before the compact mapping was added.
    """
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    qid = obj.get("qid")
    if not isinstance(qid, str) or not qid.strip():
        return None
    candidates = obj.get("candidates")
    if not isinstance(candidates, list):
        return None

    turn_ids: list[str] = []
    rebuilt_label_map: dict[str, str] = {}
    for c in candidates:
        if not isinstance(c, dict):
            continue
        label, turn_id = c.get("label"), c.get("turn_id")
        if isinstance(label, str) and label.strip() and isinstance(turn_id, str) and turn_id.strip():
            rebuilt_label_map[label] = turn_id
            turn_ids.append(turn_id)

    label_map = rebuilt_label_map
    explicit_label_map = obj.get("label_map")
    if isinstance(explicit_label_map, dict):
        cleaned = {
            k: v for k, v in explicit_label_map.items() if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
        }
        if cleaned:
            label_map = cleaned

    question = obj.get("question")
    return qid, CandidateDumpRecord(
        qid=qid,
        question=question if isinstance(question, str) else "",
        turn_ids=turn_ids,
        label_map=label_map,
    )


def load_candidate_dump(dump_dir: str | Path, conv_id: str) -> dict[str, CandidateDumpRecord]:
    """Load `<dump_dir>/<conv_id>.jsonl` (a `--dump-candidates` output, see
    `build_candidate_dump_line`) into `{qid: CandidateDumpRecord}`.

    Never raises: a missing file (the conversation hasn't been dumped, or no
    `--dump-candidates`/`--candidates-dir` was given) returns `{}`;
    malformed lines are skipped; a duplicate `qid` keeps its last
    occurrence.
    """
    path = Path(dump_dir) / f"{conv_id}.jsonl"
    records: dict[str, CandidateDumpRecord] = {}
    if not path.exists():
        return records
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for raw_line in raw_lines:
        parsed = _parse_candidate_dump_line(raw_line)
        if parsed is None:
            continue
        qid, record = parsed
        records[qid] = record  # duplicates: last one wins
    return records


def _parse_llm_ranking_line(line: str) -> tuple[str, list[str]] | None:
    """Parse one offline-LLM-ranking JSONL line
    (``{"qid", "ranked": ["c07", "c02", ...]}``). Same tolerance contract as
    `_parse_query_line`: None (never raises) for a blank line, invalid JSON,
    a non-object, a missing/blank `qid`, or a non-list `ranked`."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    qid = obj.get("qid")
    if not isinstance(qid, str) or not qid.strip():
        return None
    ranked = _clean_str_list(obj.get("ranked"))
    if ranked is None:
        return None
    return qid, ranked


def load_llm_rankings(llmrank_dir: str | Path, conv_id: str) -> dict[str, list[str]]:
    """Load `<llmrank_dir>/<conv_id>.jsonl` (an offline LLM listwise-ranking
    output, produced by other worker sessions later, P9 step D) into
    `{qid: [label, ...]}` (best first, possibly fewer labels than the
    matching dump's candidate count -- the LLM may drop candidates it judges
    irrelevant).

    Never raises: a missing file (the ranking worker hasn't started/
    finished, or `--llmrank-dir` wasn't given) returns `{}` --
    `bench.retrievers.V4LLMRerankedRetriever` falls back to the parent
    retriever's own order for any qid with no ranking here, never an error.
    Malformed lines are skipped; a duplicate `qid` keeps its last
    occurrence.
    """
    path = Path(llmrank_dir) / f"{conv_id}.jsonl"
    rankings: dict[str, list[str]] = {}
    if not path.exists():
        return rankings
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rankings
    for raw_line in raw_lines:
        parsed = _parse_llm_ranking_line(raw_line)
        if parsed is None:
            continue
        qid, ranked = parsed
        rankings[qid] = ranked  # duplicates: last one wins
    return rankings


def _load_turn_ids(turns_dir: str | Path, conv_id: str) -> list[str] | None:
    """Read `<turns_dir>/<conv_id>.json` (a JSON list of raw turn records,
    each at least `{"id": "..."}`) and return the turn ids in file order.
    Returns None (never raises) if the file is missing, unreadable, not
    valid JSON, or not a JSON list -- callers skip that conversation from
    the coverage report rather than crashing over one bad/absent snapshot.
    """
    path = Path(turns_dir) / f"{conv_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    ids = [t["id"] for t in data if isinstance(t, dict) and isinstance(t.get("id"), str)]
    return ids


def coverage_report(enrich_dir: str | Path, turns_dir: str | Path) -> dict[str, Any]:
    """Summarize enrichment coverage across every conversation found in
    `turns_dir` (each `<turns_dir>/<conv_id>.json` is the ground-truth list
    of that conversation's raw turn ids, independent of whether/how far the
    matching `<enrich_dir>/<conv_id>.jsonl` has been written).

    Tolerant like `load_enrichment`: a conversation whose `turns_dir`
    snapshot is missing/malformed is left out of `per_conversation` and
    counted in `n_conversations_skipped`, rather than raising. A
    conversation with no matching enrichment file yet still gets a
    `per_conversation` entry with `n_turns_with_facts=0` / `file_found=False`
    (partial progress across a still-running enrichment job is exactly what
    this report exists to show).
    """
    turns_dir = Path(turns_dir)
    conv_ids = sorted(p.stem for p in turns_dir.glob("*.json")) if turns_dir.exists() else []

    per_conv: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for conv_id in conv_ids:
        turn_ids = _load_turn_ids(turns_dir, conv_id)
        if turn_ids is None:
            skipped.append(conv_id)
            continue
        records, stats = _load_enrichment_with_stats(enrich_dir, conv_id)
        n_turns = len(turn_ids)
        n_with_facts = sum(1 for tid in turn_ids if records.get(tid) and records[tid].facts)
        n_facts = sum(len(r.facts) for r in records.values())
        per_conv[conv_id] = {
            "n_turns": n_turns,
            "n_turns_with_facts": n_with_facts,
            "coverage": (n_with_facts / n_turns) if n_turns else 0.0,
            "n_facts": n_facts,
            "n_enrichment_lines": stats["n_lines"],
            "n_malformed_lines": stats["n_malformed"],
            "file_found": bool(stats["file_found"]),
        }

    total_turns = sum(v["n_turns"] for v in per_conv.values())
    total_with_facts = sum(v["n_turns_with_facts"] for v in per_conv.values())
    total_facts = sum(v["n_facts"] for v in per_conv.values())
    total_malformed = sum(v["n_malformed_lines"] for v in per_conv.values())

    return {
        "enrich_dir": str(enrich_dir),
        "turns_dir": str(turns_dir),
        "n_conversations": len(per_conv),
        "n_conversations_skipped": len(skipped),
        "skipped_conversations": skipped,
        "n_conversations_with_file": sum(1 for v in per_conv.values() if v["file_found"]),
        "total_turns": total_turns,
        "total_turns_with_facts": total_with_facts,
        "overall_coverage": (total_with_facts / total_turns) if total_turns else 0.0,
        "total_facts": total_facts,
        "total_malformed_lines": total_malformed,
        "per_conversation": per_conv,
    }


def _main(argv: list[str] | None = None) -> int:
    """`python -m bench.enrich --enrich-dir ... --turns-dir ...`: print a
    one-line-per-conversation coverage summary. Convenience only -- not used
    by run_bench.py, which computes its own coverage from the conversations
    it actually indexes (see bench/retrievers.py V4EnrichedRetriever)."""
    import argparse

    default_enrich_dir = Path(__file__).resolve().parent / "data" / "enrich" / "locomo_facts"
    default_turns_dir = Path(__file__).resolve().parent / "data" / "enrich" / "locomo_turns"
    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--enrich-dir", default=str(default_enrich_dir))
    p.add_argument("--turns-dir", default=str(default_turns_dir))
    args = p.parse_args(argv)

    report = coverage_report(args.enrich_dir, args.turns_dir)
    for conv_id, stats in sorted(report["per_conversation"].items()):
        found = "yes" if stats["file_found"] else "no "
        print(
            f"{conv_id:>10}  file={found}  turns={stats['n_turns']:>4}  "
            f"with_facts={stats['n_turns_with_facts']:>4} ({stats['coverage']:.1%})  "
            f"facts={stats['n_facts']:>4}  malformed_lines={stats['n_malformed_lines']}"
        )
    print(
        f"TOTAL  conversations={report['n_conversations']} "
        f"(with_file={report['n_conversations_with_file']}, skipped={report['n_conversations_skipped']})  "
        f"turns={report['total_turns']}  with_facts={report['total_turns_with_facts']} "
        f"({report['overall_coverage']:.1%})  facts={report['total_facts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
