"""Core value types shared by the scrapers, rating engines and predictor."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

# Race "kind" drives which rating pool an entry updates. Sprint and distance
# ability are only loosely correlated, so they are rated separately.
SPRINT = "sprint"
DISTANCE = "distance"

CLASSIC = "C"
FREE = "F"
BOTH = "CF"      # skiathlon
UNKNOWN = "?"


@dataclass(frozen=True)
class Athlete:
    fis_code: str
    name: str
    nation: Optional[str] = None
    birth_year: Optional[int] = None
    competitor_id: Optional[str] = None


@dataclass(frozen=True)
class Race:
    """A single World Cup race, identified by its FIS raceid."""

    race_id: str
    season: Optional[int] = None
    event_id: Optional[str] = None
    race_date: Optional[date] = None
    place: Optional[str] = None
    series: Optional[str] = None
    title: Optional[str] = None          # e.g. "Women's 10km Interval Start Classic"
    gender: Optional[str] = None         # "M" | "W"
    kind: str = UNKNOWN                  # SPRINT | DISTANCE
    technique: str = UNKNOWN             # CLASSIC | FREE | BOTH
    start_type: Optional[str] = None     # interval | mass | pursuit | skiathlon | relay
    length_km: Optional[float] = None
    is_team: bool = False


@dataclass(frozen=True)
class Result:
    race_id: str
    fis_code: str
    rank: Optional[int]                  # None for DNF/DNS/DSQ
    bib: Optional[int] = None
    time_s: Optional[float] = None
    diff_s: Optional[float] = None
    fis_points: Optional[float] = None
    status: str = "OK"                   # OK | DNF | DNS | DSQ | LAP


@dataclass(frozen=True)
class StartEntry:
    """One athlete on a start list for a race that has not been run yet."""

    race_id: str
    fis_code: str
    bib: Optional[int] = None
    start_time: Optional[str] = None


@dataclass
class RaceEntries:
    """Whatever a race page yielded: a start list, results, or both."""

    race: Race
    athletes: List[Athlete] = field(default_factory=list)
    results: List[Result] = field(default_factory=list)
    start_list: List[StartEntry] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        return any(r.rank is not None for r in self.results)
