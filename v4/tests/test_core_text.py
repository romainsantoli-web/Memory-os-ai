"""Tokenization, sentence splitting, chunking, contextual headers, stemming."""

from memory_os_v4.store import MemoryItem
from memory_os_v4.text import (
    chunk_text,
    contextual_header,
    porter_stem,
    split_sentences,
    stem_word,
    tokenize,
)


def test_tokenize_lowercases_and_strips_accents():
    assert tokenize("Café à Paris") == ["cafe", "paris"]


def test_tokenize_keeps_code_identifiers_whole():
    toks = tokenize("Rewriting mm_cross_v2 and xgb_v14 today")
    assert "mm_cross_v2" in toks
    assert "xgb_v14" in toks


def test_tokenize_keeps_iso_dates_whole():
    toks = tokenize("Deployed on 2026-09-12 successfully")
    assert "2026-09-12" in toks


def test_tokenize_drops_short_tokens_and_stopwords():
    toks = tokenize("I am a fan of it and of the sea")
    assert "a" not in toks
    assert "the" not in toks
    assert "of" not in toks


def test_tokenize_drops_french_stopwords():
    toks = tokenize("Le chat et la souris sont dans le jardin")
    assert "le" not in toks
    assert "et" not in toks
    assert "chat" in toks
    assert "souris" in toks


def test_split_sentences_basic():
    sents = split_sentences("Hello world. How are you? Fine!")
    assert sents == ["Hello world.", "How are you?", "Fine!"]


def test_split_sentences_on_newlines():
    sents = split_sentences("Line one\nLine two\n\nLine three")
    assert len(sents) == 3


def test_chunk_text_respects_max_tokens_and_never_splits_mid_sentence():
    text = " ".join(f"This is sentence number {i} with some words." for i in range(20))
    chunks = chunk_text(text, max_tokens=30, overlap=5)
    assert len(chunks) > 1
    sentences = set(split_sentences(text))
    for chunk in chunks:
        for piece in split_sentences(chunk):
            assert piece in sentences


def test_chunk_text_hard_splits_a_single_oversized_sentence():
    long_sentence = " ".join(f"word{i}" for i in range(400)) + "."
    chunks = chunk_text(long_sentence, max_tokens=50, overlap=10)
    assert len(chunks) > 1


def test_chunk_text_overlap_shares_trailing_sentence():
    text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima."
    chunks = chunk_text(text, max_tokens=6, overlap=3)
    assert len(chunks) >= 2
    # some sentence from the first chunk should recur in the second (overlap)
    first_sents = set(split_sentences(chunks[0]))
    second_sents = set(split_sentences(chunks[1]))
    assert first_sents & second_sents


def test_chunk_text_empty_input():
    assert chunk_text("") == []


def test_chunk_text_short_sentence_then_oversized_sentence_does_not_hang():
    """Regression (adversarial review, 2026-09-12): a short sentence ahead of
    one exceeding max_tokens used to infinite-loop (the overlap carry-over
    reproduced the same state forever without advancing). Must terminate and
    must not silently drop the short sentence or any oversized-sentence word."""
    long_words = [f"word{i}" for i in range(500)]
    text = "Ok. " + " ".join(long_words) + "."
    chunks = chunk_text(text, max_tokens=180, overlap=30)
    assert len(chunks) > 1
    assert chunks[0] == "Ok."
    # every word from the oversized sentence must survive, in order, exactly once
    produced_words = " ".join(chunks[1:]).split()
    assert len(produced_words) == len(long_words)
    for original, produced in zip(long_words, produced_words):
        assert produced.startswith(original)  # last one also carries the trailing "."


def test_chunk_text_multiple_short_sentences_then_oversized_terminates():
    text = "Ok. Sure. " + " ".join(f"tok{i}" for i in range(300)) + "."
    chunks = chunk_text(text, max_tokens=50, overlap=10)
    assert len(chunks) > 1


PORTER_PAIRS = [
    # Straight from Porter's (1980) own illustrative vocabulary/output lists;
    # see docs/ABLATIONS.md E2 for the source and the two words deliberately
    # excluded (guessed, not verified against the reference implementation).
    ("caresses", "caress"),
    ("ponies", "poni"),
    ("ties", "ti"),
    ("caress", "caress"),
    ("cats", "cat"),
    ("feed", "feed"),
    ("agreed", "agre"),
    ("plastered", "plaster"),
    ("bled", "bled"),
    ("motoring", "motor"),
    ("sing", "sing"),
    ("conflated", "conflat"),
    ("troubled", "troubl"),
    ("sized", "size"),
    ("hopping", "hop"),
    ("tanned", "tan"),
    ("falling", "fall"),
    ("hissing", "hiss"),
    ("fizzed", "fizz"),
    ("failing", "fail"),
    ("happy", "happi"),
    ("relational", "relat"),
    ("national", "nation"),
    ("generalization", "gener"),
    ("revival", "reviv"),
    ("allowance", "allow"),
    ("inference", "infer"),
    ("adjustable", "adjust"),
    ("irritant", "irrit"),
    ("adoption", "adopt"),
]


def test_porter_stem_matches_20_known_pairs():
    assert len(PORTER_PAIRS) >= 20
    for word, expected in PORTER_PAIRS:
        got = porter_stem(word)
        assert got == expected, f"porter_stem({word!r}) == {got!r}, expected {expected!r}"


def test_porter_stem_is_idempotent_on_already_stemmed_words():
    for word, expected in PORTER_PAIRS[:10]:
        assert porter_stem(expected) == porter_stem(porter_stem(word))


def test_stem_word_leaves_non_alpha_tokens_untouched():
    # code identifiers / ISO dates never reach the stemmer (tokenize() only
    # ever hands stem_word what _TOKEN_RE produced: [a-z0-9_]+ or a date).
    assert stem_word("mm_cross_v2") == "mm_cross_v2"
    assert stem_word("2026-09-12") == "2026-09-12"
    assert stem_word("xgb_v14") == "xgb_v14"


def test_stem_word_french_plural_light_stripping():
    assert stem_word("chats") == "chat"
    assert stem_word("clients") == "client"


def test_stem_word_french_ement_suffix():
    assert stem_word("rapidement") == "rapid"


def test_stem_word_keeps_tion_and_sion_endings():
    assert stem_word("solution") == "solution"
    assert stem_word("decision") == "decision"


def test_tokenize_stem_true_reduces_english_plural_variants_to_same_token():
    a = tokenize("The cats are running", stem=True)
    b = tokenize("A cat was running", stem=True)
    assert "cat" in a
    assert "cat" in b


def test_tokenize_stem_false_is_unchanged_default_behaviour():
    assert tokenize("The cats are running") == tokenize("The cats are running", stem=False)
    assert "cats" in tokenize("The cats are running")


def test_contextual_header_format():
    item = MemoryItem(
        id="x", kind="episode", text="hi", role="user", project="xtx-super", event_time=1789564800.0
    )
    header = contextual_header(item)
    assert header.startswith("[")
    assert "project: xtx-super" in header
    assert "user" in header
