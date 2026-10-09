"""Date parsing (EN/FR, absolute + relative) and validity/recency helpers."""

from datetime import datetime, timezone

from memory_os_v4.store import MemoryItem
from memory_os_v4.temporal import is_valid_at, parse_absolute_dates, parse_dates, recency_weight

REF = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def _date(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def test_parse_iso_date():
    ts = parse_dates("Deployed on 2026-09-12.", REF)
    assert ts and _date(ts[0]).date() == datetime(2026, 9, 12).date()


def test_parse_dmy_slash_date():
    ts = parse_dates("Meeting on 12/09/2026 confirmed.", REF)
    assert ts
    d = _date(ts[0])
    assert (d.day, d.month, d.year) == (12, 9, 2026)


def test_parse_day_month_year_english():
    ts = parse_dates("It happened on 12 September 2026.", REF)
    assert ts
    d = _date(ts[0])
    assert (d.day, d.month, d.year) == (12, 9, 2026)


def test_parse_day_month_year_french():
    ts = parse_dates("C'est arrivé le 12 septembre 2026.", REF)
    assert ts
    d = _date(ts[0])
    assert (d.day, d.month, d.year) == (12, 9, 2026)


def test_parse_month_year_only():
    ts = parse_dates("Planned for September 2026.", REF)
    assert ts
    d = _date(ts[0])
    assert (d.month, d.year) == (9, 2026)
    assert d.day == 1


def test_parse_relative_yesterday_english():
    ts = parse_dates("We shipped it yesterday.", REF)
    assert ts
    assert abs(ts[0] - (REF - 86400)) < 2


def test_parse_relative_hier_french():
    ts = parse_dates("On a livré hier.", REF)
    assert ts
    assert abs(ts[0] - (REF - 86400)) < 2


def test_parse_relative_n_days_ago_english():
    ts = parse_dates("This started 3 days ago.", REF)
    assert ts
    assert abs(ts[0] - (REF - 3 * 86400)) < 2


def test_parse_relative_n_jours_ago_french():
    ts = parse_dates("Cela a commencé il y a 3 jours.", REF)
    assert ts
    assert abs(ts[0] - (REF - 3 * 86400)) < 2


def test_parse_relative_last_week_french():
    ts = parse_dates("On en a parlé la semaine dernière.", REF)
    assert ts
    assert abs(ts[0] - (REF - 7 * 86400)) < 2


def test_parse_no_dates_returns_empty():
    assert parse_dates("Nothing temporal here at all.", REF) == []


# ---------------------------------------------------------------------------
# A7: DD/MM default with MM/DD fallback when DD/MM is impossible
# (adversarial review, 2026-09-12)
# ---------------------------------------------------------------------------
def test_parse_dmy_ambiguous_defaults_to_ddmm():
    ts = parse_dates("on 01/02/2026", REF)
    assert ts
    d = _date(ts[0])
    assert (d.day, d.month, d.year) == (1, 2, 2026)  # DD/MM: 1 February


def test_parse_dmy_falls_back_to_mmdd_when_ddmm_impossible():
    """'03/15/2026': DD/MM would need a 15th month (invalid) -- 15 can't be
    a day-of-15th-month either way, so this can ONLY be MM/DD (March 15)."""
    ts = parse_dates("on 03/15/2026", REF)
    assert ts
    d = _date(ts[0])
    assert (d.day, d.month, d.year) == (15, 3, 2026)


def test_parse_dmy_unambiguous_ddmm_still_works():
    ts = parse_dates("Meeting on 25/12/2026 confirmed.", REF)
    assert ts
    d = _date(ts[0])
    assert (d.day, d.month, d.year) == (25, 12, 2026)  # 25 can't be a month -> stays DD/MM


def test_parse_absolute_dates_excludes_relative_expressions():
    assert parse_absolute_dates("We shipped it yesterday.") == []
    assert parse_absolute_dates("This started 3 days ago.") == []
    ts = parse_absolute_dates("Deployed on 2026-09-12.")
    assert ts and _date(ts[0]).date() == datetime(2026, 9, 12).date()


def test_parse_absolute_dates_needs_no_reference_time():
    # no reference_time parameter at all -- absolute dates are self-contained
    ts = parse_absolute_dates("It happened on 12 September 2026.")
    assert ts
    d = _date(ts[0])
    assert (d.day, d.month, d.year) == (12, 9, 2026)


def test_is_valid_at_open_interval():
    item = MemoryItem(id="a", kind="fact", text="x", valid_from=None, valid_until=None)
    assert is_valid_at(item, REF)


def test_is_valid_at_bounded_interval():
    item = MemoryItem(id="a", kind="fact", text="x", valid_from=REF, valid_until=REF + 100)
    assert not is_valid_at(item, REF - 1)
    assert is_valid_at(item, REF + 50)
    assert not is_valid_at(item, REF + 100)


def test_recency_weight_half_life():
    assert recency_weight(0, 30) == 1.0
    assert abs(recency_weight(30, 30) - 0.5) < 1e-9
    assert recency_weight(300, 30) < 0.01
