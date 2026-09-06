"""Parser tests against a real (trimmed) FIS results page.

Fixture: raceid 46733, Ruka FIN, Women's 10km Interval Start Classic,
2024-11-29, saved from fis-ski.com. Keeping a real page in the repo means a
FIS layout change shows up as a test failure rather than as silently empty
scrapes.
"""
from pathlib import Path

import pytest

from xcpredict.models import CLASSIC, DISTANCE, FREE, SPRINT, BOTH
from xcpredict.scrape import fis

FIXTURE = Path(__file__).parent / "fixtures" / "race_46733_results.html"


@pytest.fixture(scope="module")
def entries():
    return fis.parse_race_page(FIXTURE.read_text(), race_id="46733", season=2025)


def test_rows_are_parsed(entries):
    assert len(entries.results) == 12
    assert entries.has_results


def test_winner_fields(entries):
    winner = entries.results[0]
    athlete = entries.athletes[0]
    assert winner.rank == 1
    assert winner.fis_code == "3506154"
    assert winner.bib == 46
    assert winner.time_s == pytest.approx(25 * 60 + 16.2)
    assert winner.fis_points == pytest.approx(0.0)
    assert athlete.name == "KARLSSON Frida"
    assert athlete.nation == "SWE"
    assert athlete.birth_year == 1999
    assert athlete.competitor_id == "206324"


def test_time_behind_is_read_as_a_gap_not_a_time(entries):
    second = entries.results[1]
    assert second.rank == 2
    assert second.diff_s == pytest.approx(46.5)
    assert second.time_s == pytest.approx(26 * 60 + 2.7)


def test_minutes_in_the_gap_column(entries):
    """+1:03.9 is 63.9 seconds, not 1.039."""
    fifth = next(r for r in entries.results if r.rank == 5)
    assert fifth.diff_s == pytest.approx(63.9)


def test_ranks_are_dense_and_ordered(entries):
    assert [r.rank for r in entries.results] == list(range(1, 13))


def test_no_duplicate_athletes(entries):
    codes = [a.fis_code for a in entries.athletes]
    assert len(codes) == len(set(codes))


def test_race_metadata(entries):
    race = entries.race
    assert race.place == "Ruka (FIN)"
    assert race.title == "Women's 10km Interval Start Classic"
    assert race.gender == "W"
    assert race.kind == DISTANCE
    assert race.technique == CLASSIC
    assert race.start_type == "interval"
    assert race.length_km == 10.0
    assert race.is_team is False
    assert race.race_date.isoformat() == "2024-11-29"


@pytest.mark.parametrize("title,expected", [
    ("Men's Sprint Free", {"gender": "M", "kind": SPRINT, "technique": FREE}),
    ("Women's 20km Mass Start Free",
     {"gender": "W", "kind": "distance", "technique": FREE, "start_type": "mass"}),
    ("Men's Skiathlon 20km", {"technique": BOTH, "start_type": "skiathlon"}),
    ("Women's 4x7.5km Relay Classic/Free",
     {"is_team": True, "technique": BOTH, "length_km": 7.5}),
    ("Mixed Team Sprint Classic",
     {"gender": "X", "kind": SPRINT, "is_team": True, "start_type": "team_sprint"}),
])
def test_classify_title(title, expected):
    attrs = fis.classify_title(title)
    for key, value in expected.items():
        assert attrs[key] == value, key


def test_index_page_parsers():
    html = ('<a href="/DB/general/event-details.html?sectorcode=CC&eventid=55821">x</a>'
            '<a href="/DB/general/event-details.html?sectorcode=CC&eventid=55821">dup</a>'
            '<a href="/DB/general/results.html?sectorcode=CC&raceid=46733">y</a>')
    assert fis.parse_calendar(html) == ["55821"]
    assert fis.parse_event(html) == ["46733"]


def test_urls():
    assert "seasoncode=2026" in fis.calendar_url(2026)
    assert fis.race_url("46733").endswith("raceid=46733")
