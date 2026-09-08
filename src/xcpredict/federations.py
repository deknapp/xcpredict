"""Where to find out whether a skier is actually going to start.

Two kinds of source, and the difference between them matters more than either
list does.

The problem this is for: FIS publishes a start list from the same page that
later serves results, and an entry appearing there is not a promise that the
skier will start. People withdraw, get ill, are named and then pulled, or are
listed provisionally by a federation that has not finalised its team. A
prediction that includes someone who is not at the race is worse than no
prediction, because it is confidently wrong about the one thing a start-list
model exists to get right.

So a start list gets cross-checked against the federation that actually
selects the team. Where FIS and a federation disagree, the federation is the
better source — it is the body doing the selecting.

**These nine are the ones that matter**, chosen with Nathan rather than by
counting: they supply the overwhelming majority of World Cup starters, and a
federation missing from this list is a source of exactly the false entries the
check exists to catch.

**Federations are authoritative. News is early.** A federation announcement is
the body doing the selecting saying who it selected. A newspaper reporting that
someone is ill is often days ahead of that — NRK will have it before
Skiforbundet updates anything — but it is a journalist's sentence, not a
record.

So the two are used differently, and this is the design decision worth keeping:

* A **federation** disagreeing with FIS overrides FIS.
* A **news report** never removes anybody. It raises a flag for a human, and
  the prediction says the entry is doubtful. A model that quietly drops a
  skier because a headline was misparsed is worse than one that includes
  someone who did not start, because at least the second failure is visible.

Nothing here is wired up yet. A start list for a race that has not happened
cannot be tested against until the season starts, and writing scraping code
that cannot be checked against a real page is how you end up with something
that looks fine and is wrong. The URLs below are starting points for that work,
not verified endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Federation:
    code: str                 # FIS nation code, as it appears in results
    name: str
    site: str
    #: Where team selections and start lists tend to be announced. A starting
    #: point for the scraper, not a verified endpoint -- none of these have
    #: been fetched or parsed yet.
    news_hint: Optional[str] = None
    note: str = ""


FEDERATIONS: List[Federation] = [
    Federation(
        "NOR", "Norges Skiforbund", "https://www.skiforbundet.no",
        "https://www.skiforbundet.no/langrenn/",
        "The deepest team in the sport, and the one where selection is most "
        "contested -- a Norwegian named on a FIS list may still be left out.",
    ),
    Federation(
        "SWE", "Svenska Skidförbundet", "https://www.skidor.com",
        "https://www.skidor.com/grenar/langdskidor/",
    ),
    Federation(
        "FIN", "Suomen Hiihtoliitto", "https://www.hiihtoliitto.fi",
        "https://www.hiihtoliitto.fi/maastohiihto/",
    ),
    Federation(
        "ITA", "FISI", "https://www.fisi.org",
        "https://www.fisi.org/sci-di-fondo/",
    ),
    Federation(
        "FRA", "Fédération Française de Ski", "https://www.ffs.fr",
        "https://www.ffs.fr/ski-de-fond",
    ),
    Federation(
        "GER", "Deutscher Skiverband", "https://www.deutscherskiverband.de",
        "https://www.deutscherskiverband.de/langlauf/",
    ),
    Federation(
        "USA", "U.S. Ski & Snowboard", "https://usskiandsnowboard.org",
        "https://usskiandsnowboard.org/sport-development/cross-country",
        "Also worth watching for late additions: US starts are sometimes "
        "confirmed close to the race.",
    ),
    Federation(
        "SUI", "Swiss-Ski", "https://www.swiss-ski.ch",
        "https://www.swiss-ski.ch/langlauf/",
    ),
    Federation(
        "CAN", "Nordiq Canada", "https://nordiqcanada.ca",
        "https://nordiqcanada.ca/discipline/cross-country/",
    ),
]


@dataclass(frozen=True)
class NewsSource:
    name: str
    nation: str
    site: str
    section: str
    language: str
    note: str = ""


#: Outlets that cover cross-country closely enough to report an illness or a
#: withdrawal before it reaches an official list.
#:
#: Weighted toward Scandinavia on purpose rather than for balance: that is
#: where most of the field comes from, and where the coverage is daily and
#: detailed enough to be worth parsing. NRK in particular reports team news
#: several days ahead of the federation.
NEWS_SOURCES: List[NewsSource] = [
    NewsSource(
        "NRK Sport", "NOR", "https://www.nrk.no",
        "https://www.nrk.no/sport/langrenn/", "no",
        "The single most valuable source here. Covers Norwegian team news "
        "daily and reports illness and withdrawals well before Skiforbundet.",
    ),
    NewsSource(
        "SVT Sport", "SWE", "https://www.svt.se",
        "https://www.svt.se/sport/langdskidor/", "sv",
        "The Swedish equivalent of NRK for reliability.",
    ),
    NewsSource(
        "Expressen Sport", "SWE", "https://www.expressen.se",
        "https://www.expressen.se/sport/langdskidor/", "sv",
        "Faster than SVT and less careful. Useful as an early signal, worth "
        "less as confirmation.",
    ),
    NewsSource(
        "Yle Urheilu", "FIN", "https://yle.fi",
        "https://yle.fi/urheilu/hiihto", "fi",
    ),
    NewsSource(
        "FIS Cross-Country", "INT", "https://www.fis-ski.com",
        "https://www.fis-ski.com/DB/cross-country/", "en",
        "Official but slow. Included as a backstop, not a leading indicator.",
    ),
]

BY_CODE = {f.code: f for f in FEDERATIONS}
NEWS_BY_NATION: dict = {}
for _s in NEWS_SOURCES:
    NEWS_BY_NATION.setdefault(_s.nation, []).append(_s)


#: How a signal about a skier should be treated. The ordering is the point:
#: nothing below "federation" may remove an entry on its own.
CONFIDENCE = {
    "fis_startlist": 1.0,     # the official entry
    "federation": 1.0,        # overrides FIS where they disagree
    "news_confirmed": 0.6,    # a report naming the skier and the race
    "news_rumour": 0.3,       # a report that is vaguer than that
}


def treat(signal: str) -> str:
    """What the pipeline is allowed to do with a signal of this kind."""
    if signal in ("fis_startlist", "federation"):
        return "authoritative"
    return "flag_for_review"


def covers(nation_code: Optional[str]) -> bool:
    """Whether a start-list entry can be cross-checked against a federation."""
    return nation_code in BY_CODE


def uncovered(nation_codes) -> List[str]:
    """Nations in a field that no federation on this list can verify.

    Reported rather than ignored: a start list is only as trustworthy as its
    least-checked entry, and a prediction should be able to say how much of its
    field was actually confirmed.
    """
    return sorted({n for n in nation_codes if n and n not in BY_CODE})
