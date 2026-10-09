"""Date parsing (absolute + relative, EN/FR) and recency/validity helpers.

No dateutil dependency: month arithmetic for "last/next month" is done by hand.
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryItem

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "août": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12, "décembre": 12,
}
_MONTH_NAMES = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
    "saturday": 5, "sunday": 6,
    "lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4,
    "samedi": 5, "dimanche": 6,
}

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:de\s+)?({_MONTH_NAMES})\s+(\d{{4}})\b", re.IGNORECASE
)
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_NAMES})\s+(\d{{4}})\b", re.IGNORECASE)

_N_DAYS_AGO_EN_RE = re.compile(r"\b(\d+)\s+(day|days|week|weeks|month|months)\s+ago\b", re.IGNORECASE)
_N_DAYS_AGO_FR_RE = re.compile(
    r"\bil y a\s+(\d+)\s+(jour|jours|semaine|semaines|mois)\b", re.IGNORECASE
)
_NEXT_WEEKDAY_RE = re.compile(
    r"\bnext\s+(" + "|".join(_WEEKDAYS.keys()) + r")\b", re.IGNORECASE
)


def _add_months(dt: datetime, delta: int) -> datetime:
    month_index = dt.month - 1 + delta
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _to_ts(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_absolute_dates(text: str) -> list[float]:
    """Parse only ABSOLUTE date expressions (ISO, D/M/Y, "12 September 2026",
    "September 2026") into unix timestamps -- no ``reference_time`` needed,
    unlike relative expressions ("yesterday", "3 days ago").

    Two-field slash dates (``_DMY_RE``, e.g. "12/09/2026") default to
    **DD/MM/YYYY** (day first) -- the convention this module's EN/FR mixed
    corpus mostly uses -- but fall back to MM/DD when DD/MM is impossible:
    if the field DD/MM would read as the month is > 12 (not a valid month)
    while swapping the two fields gives a valid date, that swapped (MM/DD)
    reading is used instead (e.g. "03/15/2026" -> March 15, not the 3rd of
    an invalid 15th month). Both fields <= 12 stays ambiguous and keeps the
    DD/MM default (e.g. "01/02/2026" -> 1 February, not January 2nd).

    Used by :func:`parse_dates` (below) and by
    ``extract.normalize_client_facts`` (A7, docs/ABLATIONS.md "Hardening
    2026-09-13"): a fact's own text can supply its ``event_time`` without
    needing a reference time, which relative expressions inherently require.
    """
    if not text:
        return []
    results: list[float] = []

    for y, m, d in _ISO_RE.findall(text):
        try:
            results.append(_to_ts(datetime(int(y), int(m), int(d), tzinfo=timezone.utc)))
        except ValueError:
            continue

    for a, b, y in _DMY_RE.findall(text):
        day, month = int(a), int(b)
        if month > 12 and day <= 12:
            day, month = month, day  # MM/DD fallback -- see docstring
        try:
            results.append(_to_ts(datetime(int(y), month, day, tzinfo=timezone.utc)))
        except ValueError:
            continue

    for d, month_name, y in _DAY_MONTH_YEAR_RE.findall(text):
        month = _MONTHS.get(month_name.lower())
        if month:
            try:
                results.append(_to_ts(datetime(int(y), month, int(d), tzinfo=timezone.utc)))
            except ValueError:
                continue

    day_month_year_spans = {m.span() for m in _DAY_MONTH_YEAR_RE.finditer(text)}
    for m in _MONTH_YEAR_RE.finditer(text):
        if any(a <= m.start() and m.end() <= b for a, b in day_month_year_spans):
            continue  # already captured with a day
        month = _MONTHS.get(m.group(1).lower())
        if month:
            results.append(_to_ts(datetime(int(m.group(2)), month, 1, tzinfo=timezone.utc)))

    return results


def parse_dates(text: str, reference_time: float) -> list[float]:
    """Parse absolute and relative date expressions (EN/FR) into unix timestamps.

    ``reference_time`` anchors relative expressions ("yesterday", "hier",
    "3 days ago", "il y a 3 jours", "last week", "la semaine dernière",
    "next Monday"). See :func:`parse_absolute_dates` for the DD/MM vs MM/DD
    convention used for two-field slash dates.
    """
    if not text:
        return []
    ref = datetime.fromtimestamp(reference_time, tz=timezone.utc)
    lower = text.lower()
    results: list[float] = list(parse_absolute_dates(text))

    for n, unit in _N_DAYS_AGO_EN_RE.findall(lower):
        results.append(_to_ts(_shift(ref, -int(n), unit)))
    for n, unit in _N_DAYS_AGO_FR_RE.findall(lower):
        results.append(_to_ts(_shift(ref, -int(n), unit)))

    if re.search(r"\byesterday\b", lower) or re.search(r"\bhier\b", lower):
        results.append(_to_ts(ref - timedelta(days=1)))
    if re.search(r"\btoday\b", lower) or re.search(r"\baujourd'?hui\b", lower):
        results.append(_to_ts(ref))
    if re.search(r"\btomorrow\b", lower) or re.search(r"\bdemain\b", lower):
        results.append(_to_ts(ref + timedelta(days=1)))

    if re.search(r"\blast week\b", lower) or re.search(r"la semaine derni\w*re", lower):
        results.append(_to_ts(ref - timedelta(days=7)))
    if re.search(r"\bnext week\b", lower) or re.search(r"la semaine prochaine", lower):
        results.append(_to_ts(ref + timedelta(days=7)))
    if re.search(r"\blast month\b", lower) or re.search(r"le mois dernier", lower):
        results.append(_to_ts(_add_months(ref, -1)))
    if re.search(r"\bnext month\b", lower) or re.search(r"le mois prochain", lower):
        results.append(_to_ts(_add_months(ref, 1)))

    m = _NEXT_WEEKDAY_RE.search(lower)
    if m:
        target = _WEEKDAYS[m.group(1).lower()]
        days_ahead = (target - ref.weekday() - 1) % 7 + 1
        results.append(_to_ts(ref + timedelta(days=days_ahead)))

    return results


def _shift(ref: datetime, n: int, unit: str) -> datetime:
    unit = unit.lower()
    if unit.startswith(("day", "jour")):
        return ref + timedelta(days=n)
    if unit.startswith(("week", "semaine")):
        return ref + timedelta(weeks=n)
    if unit.startswith(("month", "mois")):
        return _add_months(ref, n)
    return ref


def is_valid_at(item: "MemoryItem", t: float) -> bool:
    """True if ``item`` is within its bi-temporal validity window at time ``t``."""
    if item.valid_from is not None and t < item.valid_from:
        return False
    if item.valid_until is not None and t >= item.valid_until:
        return False
    return True


def recency_weight(age_days: float, half_life_days: float) -> float:
    """Exponential decay: 1.0 at age 0, 0.5 at ``half_life_days``."""
    if half_life_days <= 0:
        return 1.0 if age_days <= 0 else 0.0
    return math.exp(-math.log(2) * max(0.0, age_days) / half_life_days)
