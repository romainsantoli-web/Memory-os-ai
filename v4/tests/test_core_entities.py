"""Entity extraction (English + French) and the EntityIndex retriever."""

from memory_os_v4.entities import EntityIndex, extract_entities


def test_extracts_capitalized_name_mid_sentence():
    ents = extract_entities("We met with Marie Curie yesterday.")
    assert "marie curie" in ents


def test_extracts_allcaps_tokens():
    ents = extract_entities("The signal comes from NQ and ES futures via MBO feed.")
    assert "nq" in ents
    assert "es" in ents
    assert "mbo" in ents


def test_extracts_code_identifiers():
    ents = extract_entities("We rewrote mm_cross_v2 and updated engine.py before shipping xgb_v14.")
    assert "mm_cross_v2" in ents
    assert "engine.py" in ents
    assert "xgb_v14" in ents


def test_extracts_quoted_phrases():
    ents = extract_entities('The plan is called "Project Falcon" internally.')
    assert "project falcon" in ents


def test_extracts_numbers_with_units():
    ents = extract_entities("Latency improved by 12ms and cost dropped 15%.")
    assert any("12" in e for e in ents)
    assert any("15" in e for e in ents)


def test_extracts_iso_dates():
    ents = extract_entities("The migration ran on 2026-09-12 without issues.")
    assert "2026-09-12" in ents


def test_single_sentence_subject_at_start_is_kept():
    # Atomic facts are usually one sentence; the sentence-start subject must
    # still be captured so entity-overlap supersession can work.
    ents = extract_entities("Romain lives in Paris.")
    assert "romain" in ents
    assert "paris" in ents


def test_multi_sentence_start_word_needs_corroboration():
    text = "The plan failed. We tried again with Marie Curie and it worked."
    ents = extract_entities(text)
    # "The" is a sentence-start capitalized word with no mid-sentence echo.
    assert "the" not in ents
    assert "marie curie" in ents


def test_french_capitalized_entity():
    ents = extract_entities("Nous avons rencontré Jean Dupont à la réunion de suivi.")
    assert "jean dupont" in ents


def test_french_accents_normalized_in_entities():
    ents = extract_entities('La ville de "Genève" a été choisie pour le siège.')
    assert "geneve" in ents


def test_entity_index_search_ranks_by_overlap():
    idx = EntityIndex()
    idx.add("d1", ["nq", "es", "mbo"])
    idx.add("d2", ["nq"])
    idx.add("d3", ["lyon"])
    # padding so df/n for "nq"/"es" stays under the 10% commonality cutoff
    # (adversarial review, 2026-09-12) — this test is about overlap ranking,
    # not about that cutoff, which has its own dedicated tests.
    for i in range(30):
        idx.add(f"filler{i}", ["other"])
    results = idx.search(["nq", "es"], top_k=3)
    ids = [d for d, _ in results]
    assert ids[0] == "d1"
    assert "d3" not in ids


def test_entity_index_remove():
    idx = EntityIndex()
    idx.add("d1", ["paris"])
    idx.remove("d1")
    assert idx.search(["paris"], top_k=5) == []


def test_entity_index_persistence_roundtrip(tmp_path):
    idx = EntityIndex()
    idx.add("d1", ["nq", "es"])
    idx.add("d2", ["nq"])
    path = tmp_path / "entities.json"
    idx.save(path)
    loaded = EntityIndex.load(path)
    assert loaded.search(["nq"], top_k=5) == idx.search(["nq"], top_k=5)


def test_entity_index_search_breaks_ties_deterministically_by_id():
    """Adversarial review, 2026-09-12: `bucket` is a set, so before the fix
    ties were broken by PYTHONHASHSEED-dependent set-iteration order. Query
    two entities that give d1/d2/d3 an identical combined score and check
    the tie is always resolved the same way (ascending doc_id), run twice
    with freshly-built (differently-ordered-internally) indexes."""
    for _ in range(5):
        idx = EntityIndex()
        idx.add("zzz", ["shared"])
        idx.add("aaa", ["shared"])
        idx.add("mmm", ["shared"])
        for i in range(40):  # dilute df/n well below the 10% commonality cutoff
            idx.add(f"filler{i}", ["other"])
        result = idx.search(["shared"], top_k=5)
        ids = [d for d, _ in result]
        assert len(ids) == 3
        assert ids == sorted(ids)  # deterministic ascending tiebreak, every time


def test_entity_index_search_skips_overly_common_entities():
    """Adversarial review, 2026-09-12: an entity in >10% of the corpus (e.g.
    a dialogue's main participant, mentioned in most turns) is not
    discriminative and must not contribute score at all."""
    idx = EntityIndex()
    for i in range(20):
        idx.add(f"d{i}", ["caroline"])  # present in 100% of docs
    idx.add("d0", ["caroline", "rare_topic"])
    result = idx.search(["caroline", "rare_topic"], top_k=5)
    assert result  # rare_topic alone still yields a result
    assert result[0][0] == "d0"
    # a query on the overly-common entity alone yields nothing (fully skipped)
    assert idx.search(["caroline"], top_k=5) == []


def test_extract_entities_drops_junk_question_words():
    """Adversarial review, 2026-09-12: a query is almost always one
    sentence, so the single-sentence branch of `_capitalized_spans` used to
    keep sentence-start capitals unconditionally — turning "What did
    Caroline research?" into entities ["what", "caroline"] and polluting
    every query with a near-universal non-entity."""
    ents = extract_entities("What did Caroline research?")
    assert "what" not in ents
    assert "caroline" in ents
    ents2 = extract_entities("How did the plan work out? Wow, thanks!")
    assert "how" not in ents2
    assert "wow" not in ents2
    assert "thanks" not in ents2


# ---------------------------------------------------------------------------
# score_doc (Config.full_candidate_scoring, docs/ABLATIONS.md "Full-candidate
# scoring 2026-09-13"): must equal the score `search()` computes for the
# same doc, but work for a doc outside `search()`'s own top-k too.
# ---------------------------------------------------------------------------
def test_entity_index_score_doc_matches_search_score_for_same_doc():
    idx = EntityIndex()
    idx.add("d1", ["nq", "es", "mbo"])
    idx.add("d2", ["nq"])
    idx.add("d3", ["lyon"])
    for i in range(30):  # keep df/n under the 10% commonality cutoff
        idx.add(f"filler{i}", ["other"])
    q = ["nq", "es"]
    search_scores = dict(idx.search(q, top_k=10))
    for doc_id in ("d1", "d2", "d3"):
        assert idx.score_doc(q, doc_id) == search_scores.get(doc_id, 0.0)


def test_entity_index_score_doc_matches_search_outside_a_tiny_top_k():
    idx = EntityIndex()
    idx.add("d1", ["nq", "es", "mbo"])
    idx.add("d2", ["nq"])
    for i in range(30):
        idx.add(f"filler{i}", ["other"])
    q = ["nq", "es"]
    full_search = dict(idx.search(q, top_k=10))
    top1 = dict(idx.search(q, top_k=1))
    dropped = next(d for d in full_search if d not in top1)
    assert idx.score_doc(q, dropped) == full_search[dropped]
    assert idx.score_doc(q, dropped) > 0.0


def test_entity_index_score_doc_respects_common_entity_skip():
    """Same >10%-of-corpus skip as `search()` (see
    test_entity_index_search_skips_overly_common_entities)."""
    idx = EntityIndex()
    for i in range(20):
        idx.add(f"d{i}", ["caroline"])
    idx.add("d0", ["caroline", "rare_topic"])
    assert idx.score_doc(["caroline"], "d0") == 0.0  # fully skipped, not just downweighted
    assert idx.score_doc(["caroline", "rare_topic"], "d0") > 0.0


def test_entity_index_score_doc_unknown_doc_returns_zero():
    idx = EntityIndex()
    idx.add("d1", ["paris"])
    assert idx.score_doc(["paris"], "does-not-exist") == 0.0


def test_entity_index_score_doc_empty_query_returns_zero():
    idx = EntityIndex()
    idx.add("d1", ["paris"])
    assert idx.score_doc([], "d1") == 0.0


def test_entity_index_score_doc_no_overlap_is_zero():
    idx = EntityIndex()
    idx.add("d1", ["paris"])
    idx.add("d2", ["lyon"])
    assert idx.score_doc(["lyon"], "d1") == 0.0
