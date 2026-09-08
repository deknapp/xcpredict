"""Tests for the start-list verification sources.

Small, but they pin the one decision that matters: a news report must never be
able to remove a skier on its own. Quietly dropping someone because a headline
was misparsed is a worse failure than including someone who did not start —
the second is visible in the result, the first is not.
"""
from __future__ import annotations

from xcpredict import federations as fed


def test_the_nine_agreed_federations_are_present():
    assert {f.code for f in fed.FEDERATIONS} == {
        "NOR", "SWE", "FIN", "ITA", "FRA", "GER", "USA", "SUI", "CAN"
    }


def test_every_federation_has_somewhere_to_look():
    for f in fed.FEDERATIONS:
        assert f.site.startswith("https://")
        assert f.news_hint is None or f.news_hint.startswith("https://")


def test_covered_and_uncovered_nations_are_distinguished():
    assert fed.covers("NOR")
    assert not fed.covers("KAZ")
    assert not fed.covers(None)


def test_uncovered_reports_what_cannot_be_checked():
    """A start list is only as trustworthy as its least-checked entry."""
    assert fed.uncovered(["NOR", "KAZ", "JPN", "SWE", None]) == ["JPN", "KAZ"]


def test_news_sources_cover_the_scandinavian_nations():
    nations = {s.nation for s in fed.NEWS_SOURCES}
    assert {"NOR", "SWE", "FIN"} <= nations


def test_nrk_is_listed_for_norway():
    """Named because it reports team news days ahead of the federation."""
    assert any(s.name.startswith("NRK") for s in fed.NEWS_BY_NATION["NOR"])


def test_official_sources_are_authoritative():
    assert fed.treat("fis_startlist") == "authoritative"
    assert fed.treat("federation") == "authoritative"


def test_news_can_only_raise_a_flag():
    """The whole point. News never removes anybody by itself."""
    assert fed.treat("news_confirmed") == "flag_for_review"
    assert fed.treat("news_rumour") == "flag_for_review"


def test_an_unknown_signal_is_not_trusted():
    assert fed.treat("someone_said_so") == "flag_for_review"


def test_confidence_ranks_official_above_reported():
    assert fed.CONFIDENCE["federation"] > fed.CONFIDENCE["news_confirmed"]
    assert fed.CONFIDENCE["news_confirmed"] > fed.CONFIDENCE["news_rumour"]
