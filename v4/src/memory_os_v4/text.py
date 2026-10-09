"""Text utilities: tokenization, sentence splitting, chunking, contextual headers.

No external NLP dependency: everything here is regex/unicodedata based, tuned to
keep code identifiers (``mm_cross_v2``), dates (``2026-09-12``) and numeric tokens
intact while remaining accent- and case-insensitive for matching.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryItem

_STOPWORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on",
    "at", "to", "for", "and", "or", "but", "with", "as", "by", "this", "that", "it", "its",
    "from", "has", "have", "had", "i", "you", "he", "she", "we", "they", "my", "your", "his",
    "her", "our", "their", "not", "no", "do", "does", "did", "will", "would", "can", "could",
    "should", "so", "if", "than", "then", "there", "here", "what", "which", "who", "whom",
    "am", "into", "about", "up", "out", "over", "under",
}

_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais", "donc", "or", "ni",
    "car", "ce", "cet", "cette", "ces", "il", "elle", "ils", "elles", "nous", "vous", "je",
    "tu", "on", "que", "qui", "quoi", "dont", "ou", "a", "au", "aux", "en", "dans", "sur",
    "pour", "par", "avec", "sans", "ne", "pas", "plus", "tres", "etre", "avoir", "est", "sont",
    "etait", "etaient", "sera", "seront", "se", "sa", "son", "ses", "leur", "leurs",
}

STOPWORDS = _STOPWORDS_EN | _STOPWORDS_FR

# ISO date as one token, else alnum runs optionally joined by underscores (keeps
# identifiers like mm_cross_v2, xgb_v14 whole).
_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[a-z0-9]+(?:_[a-z0-9]+)*")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def strip_accents(s: str) -> str:
    """Strip diacritics via NFKD decomposition (café -> cafe)."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Lowercase, accent-strip, and split into tokens for BM25/entity matching.

    Keeps code identifiers (``mm_cross_v2``) and ISO dates (``2026-09-12``) as
    single tokens; drops tokens shorter than 2 characters and stopwords.

    ``stem=True`` (E2 ablation, ``config.use_stemming``) additionally reduces
    each purely-alphabetic token to its stem via :func:`stem_word` (Porter for
    English, light suffix stripping for French); code identifiers, dates and
    numbers are never stemmed (they are not ``str.isalpha()``). Display text
    is never affected — this only changes what BM25/entity matching sees.
    """
    if not text:
        return []
    normalized = strip_accents(text.lower())
    tokens = _TOKEN_RE.findall(normalized)
    tokens = [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]
    if stem:
        tokens = [stem_word(t) for t in tokens]
    return tokens


# ---------------------------------------------------------------------------
# Stemming (E2 ablation): in-house Porter stemmer (Porter, 1980) for English,
# plus a light French suffix stripper. No NLTK/snowballstemmer dependency.
# ---------------------------------------------------------------------------
def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        if i == 0:
            return True
        return not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's `m`: number of consonant-sequence -> vowel-sequence transitions."""
    form = "".join("c" if _is_consonant(stem, i) else "v" for i in range(len(stem)))
    return form.count("vc")


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    """True if `stem` ends consonant-vowel-consonant, last consonant not w/x/y."""
    if len(stem) < 3:
        return False
    return (
        _is_consonant(stem, len(stem) - 3)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 1)
        and stem[-1] not in "wxy"
    )


# (suffix, replacement); step 2/3 apply the first match whose stem has m > 0.
_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]
# step 4: bare suffix removal when m(stem) > 1; "ion" additionally requires
# the stem end in s/t (Porter's special case).
_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def porter_stem(word: str) -> str:
    """Classic Porter (1980) stemmer for English. Pure ASCII lowercase in/out."""
    word = word.lower()
    if len(word) <= 2:
        return word

    # step 1a: plurals
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # step 1b: -eed/-ed/-ing
    did_ed_or_ing = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        stem = None
        if word.endswith("ed") and _contains_vowel(word[:-2]):
            stem = word[:-2]
        elif word.endswith("ing") and _contains_vowel(word[:-3]):
            stem = word[:-3]
        if stem is not None:
            word = stem
            did_ed_or_ing = True

    if did_ed_or_ing:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"

    # step 1c: -y -> -i (if preceded by a vowel elsewhere in the stem)
    if word.endswith("y") and len(word) > 1 and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    # step 2: derivational suffixes, single longest-first match
    for suf, rep in _STEP2:
        if word.endswith(suf):
            stem = word[: -len(suf)]
            if _measure(stem) > 0:
                word = stem + rep
            break

    # step 3
    for suf, rep in _STEP3:
        if word.endswith(suf):
            stem = word[: -len(suf)]
            if _measure(stem) > 0:
                word = stem + rep
            break

    # step 4
    for suf in _STEP4:
        if word.endswith(suf):
            stem = word[: -len(suf)]
            if suf == "ion":
                if stem.endswith(("s", "t")) and _measure(stem) > 1:
                    word = stem
                break
            if _measure(stem) > 1:
                word = stem
            break

    # step 5a: drop trailing -e
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem

    # step 5b: -ll -> -l
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]

    return word


def _light_french_stem(word: str) -> str:
    """Conservative French suffix stripping: plural -s/-x and adverbial -ement.

    Deliberately leaves ``-tion``/``-sion`` untouched (a real, stable word
    ending in French, not a suffix to strip) and only fires on words long
    enough that the result cannot degenerate into noise.
    """
    if word.endswith(("tion", "sion")):
        return word
    if len(word) > 6 and word.endswith("ement"):
        return word[:-5]
    if len(word) > 4 and word.endswith(("s", "x")) and not word.endswith("ss"):
        return word[:-1]
    return word


def stem_word(word: str) -> str:
    """Stem one token: light French stripping first, else the Porter stemmer.

    Non-alphabetic tokens (code identifiers with ``_``, ISO dates, numbers —
    see ``_TOKEN_RE``) are returned unchanged: stemming only ever touches
    ordinary words, never identifiers/dates that must match exactly. Words
    already ending in ``-tion``/``-sion`` are also left untouched: Porter's
    own step-4 ``ion`` rule would otherwise strip them (``solution`` ->
    ``solut``), which the task's "tion kept" requirement rules out for the
    mixed EN/FR corpus this stemmer serves.
    """
    if not word.isalpha():
        return word
    if word.endswith(("tion", "sion")):
        return word
    fr = _light_french_stem(word)
    if fr != word:
        return fr
    return porter_stem(word)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ./!/?/… followed by whitespace, or newlines."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _sentence_len(sentence: str) -> int:
    return max(1, len(tokenize(sentence)))


def chunk_text(text: str, max_tokens: int = 180, overlap: int = 30) -> list[str]:
    """Sentence-aware chunking: never cuts mid-sentence unless one sentence alone
    exceeds ``max_tokens`` (then it is hard-split on whitespace). Consecutive
    chunks share up to ``overlap`` tokens worth of trailing sentences.

    Fix (adversarial review, 2026-09-12): an oversized sentence is now
    hard-split unconditionally, flushing any pending ``current`` buffer
    first, rather than only when ``current`` was already empty. The old
    ``slen > max_tokens and not current`` guard meant a short sentence
    (e.g. "Ok.") ahead of one oversized sentence left ``current`` non-empty;
    the oversized sentence then only ever hit the "flush + carry overlap"
    branch below, whose overlap carry-over (the same short sentence, since
    it is shorter than ``overlap``) reproduced the exact same state on the
    next loop iteration without advancing `i` — an infinite loop.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        slen = _sentence_len(sentence)

        if slen > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            words = sentence.split()
            step = max(1, max_tokens)
            for j in range(0, len(words), step):
                chunks.append(" ".join(words[j : j + step]))
            i += 1
            continue

        if current and current_len + slen > max_tokens:
            chunks.append(" ".join(current))
            overlap_sents: list[str] = []
            ov = 0
            for prev in reversed(current):
                pl = _sentence_len(prev)
                if ov + pl > overlap:
                    break
                overlap_sents.insert(0, prev)
                ov += pl
            current = overlap_sents
            current_len = ov
            continue

        current.append(sentence)
        current_len += slen
        i += 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def date_str(ts: float | None) -> str:
    if ts is None:
        return "unknown-date"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def contextual_header(item: "MemoryItem") -> str:
    """Contextual header prepended (not stored) before embedding/BM25 indexing.

    e.g. ``"[2026-09-12 | project: xtx-super | user]"``.
    """
    return f"[{date_str(item.event_time)} | project: {item.project} | {item.role}]"
