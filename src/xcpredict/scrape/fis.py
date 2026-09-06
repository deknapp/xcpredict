"""Scraper for fis-ski.com cross-country pages.

Layout notes verified against live pages (see README):

* There is no separate start-list endpoint. FIS serves the start list from the
  same ``results.html?raceid=`` page before the race is run and replaces it
  with results afterwards, so one parser covers both.
* Result rows repeat the finish time in two cells (a desktop and a
  small-screen variant) and the column order shifts between race formats, so
  cells are identified by their *content*, not by their position.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Dict, Iterator, List, Optional, Tuple

from bs4 import BeautifulSoup

from ..models import (BOTH, CLASSIC, DISTANCE, FREE, SPRINT, UNKNOWN, Athlete,
                      Race, RaceEntries, Result, StartEntry)
from .http import Fetcher

log = logging.getLogger(__name__)

BASE = "https://www.fis-ski.com/DB/general"
SECTOR = "CC"

STATUS_TOKENS = {"DNF", "DNS", "DSQ", "DQ", "LAP", "OOT", "RAL"}

RE_FIS_CODE = re.compile(r"^\d{6,8}$")
RE_YEAR = re.compile(r"^(?:19|20)\d{2}$")
RE_NATION = re.compile(r"^[A-Z]{3}$")
RE_CLOCK = re.compile(r"^\+?\d{1,3}:\d{2}(?:\.\d+)?$")
RE_SECONDS = re.compile(r"^\+\d{1,3}(?:\.\d+)?$")
RE_POINTS = re.compile(r"^\d+\.\d{2}$")
RE_INT = re.compile(r"^\d{1,3}$")
RE_DATE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


# --------------------------------------------------------------------------- URLs

def calendar_url(season: int) -> str:
    """`season` is the FIS season code: 2025 means the 2024/25 winter."""
    return (f"{BASE}/calendar-results.html?sectorcode={SECTOR}&seasoncode={season}"
            f"&categorycode=WC")


def event_url(event_id: str, season: int) -> str:
    return f"{BASE}/event-details.html?sectorcode={SECTOR}&eventid={event_id}&seasoncode={season}"


def race_url(race_id: str) -> str:
    return f"{BASE}/results.html?sectorcode={SECTOR}&raceid={race_id}"


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # lxml not installed
        return BeautifulSoup(html, "html.parser")


# ----------------------------------------------------------------- index pages

def parse_calendar(html: str) -> List[str]:
    """Event ids listed on a season calendar page, in page order."""
    return _unique(re.findall(r"event-details\.html\?[^\"']*eventid=(\d+)", html))


def parse_event(html: str) -> List[str]:
    """Race ids listed on an event page, in page order."""
    return _unique(re.findall(r"results\.html\?[^\"']*raceid=(\d+)", html))


def _unique(values: List[str]) -> List[str]:
    seen, out = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


# ------------------------------------------------------------------ race pages

def classify_title(title: str) -> Dict[str, object]:
    """Derive structured race attributes from the FIS race title.

    e.g. "Women's 10km Interval Start Classic" ->
         gender W, distance race, classic, interval start, 10.0 km.
    """
    low = (title or "").lower()

    if "women" in low or "ladies" in low:
        gender = "W"
    elif "mixed" in low:
        gender = "X"
    elif "men" in low:
        gender = "M"
    else:
        gender = None

    is_team = "relay" in low or "team sprint" in low
    kind = SPRINT if "sprint" in low else DISTANCE

    has_c = "classic" in low
    has_f = "free" in low or "skating" in low
    if "skiathlon" in low or (has_c and has_f):
        technique = BOTH
    elif has_c:
        technique = CLASSIC
    elif has_f:
        technique = FREE
    else:
        technique = UNKNOWN

    for token, start_type in (("skiathlon", "skiathlon"), ("relay", "relay"),
                              ("team sprint", "team_sprint"), ("pursuit", "pursuit"),
                              ("mass start", "mass"), ("interval", "interval"),
                              ("sprint", "sprint")):
        if token in low:
            break
    else:
        start_type = None

    # "4x7.5km" -> per-leg 7.5; "10 km" -> 10.0
    match = re.search(r"(?:\d+\s*x\s*)?(\d+(?:\.\d+)?)\s*km", low)
    length_km = float(match.group(1)) if match else None

    return {"gender": gender, "kind": kind, "technique": technique,
            "start_type": start_type, "length_km": length_km, "is_team": is_team}


def _header_text(soup: BeautifulSoup, selector: str) -> Optional[str]:
    node = soup.select_one(selector)
    return node.get_text(" ", strip=True) if node else None


def _header_date(soup: BeautifulSoup) -> Optional[date]:
    """Race date.

    FIS renders it as a `timezone-date` div carrying machine-readable
    attributes, which is worth preferring over the localised display text.
    """
    node = soup.select_one(".timezone-date[data-date]")
    if node:
        try:
            return datetime.strptime(node["data-date"][:10], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            pass
    for candidate in soup.select("[class*=event-header]"):
        text = candidate.get_text(" ", strip=True)
        if RE_DATE.match(text):
            try:
                return datetime.strptime(text, "%B %d, %Y").date()
            except ValueError:
                continue
    return None


def parse_race_header(html: str, race_id: str, season: Optional[int] = None,
                      event_id: Optional[str] = None) -> Race:
    soup = _soup(html)
    title = _header_text(soup, ".event-header__kind")
    attrs = classify_title(title or "")
    return Race(
        race_id=race_id,
        season=season,
        event_id=event_id,
        race_date=_header_date(soup),
        # .event-header__name wraps both the venue heading and the subtitle,
        # so take the heading itself rather than the whole block.
        place=_header_text(soup, ".event-header__name h1")
        or _header_text(soup, ".event-header__name"),
        series=_header_text(soup, ".event-header__subtitle"),
        title=title,
        **attrs,  # type: ignore[arg-type]
    )


def _row_cells(row) -> List[str]:
    """Visible text of the row's leaf cells, in document order.

    Leaf divs only: an outer grid cell that wraps another div contributes
    nothing of its own. Comments are dropped by the parser.
    """
    cells = []
    for div in row.find_all("div"):
        if div.find("div") is not None:
            continue
        text = div.get_text(" ", strip=True)
        if text:
            cells.append(re.sub(r"\s+", " ", text))
    return cells


def _clock_to_seconds(text: str) -> Optional[float]:
    text = text.lstrip("+")
    if ":" in text:
        parts = text.split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return None
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    try:
        return float(text)
    except ValueError:
        return None


def parse_row(row, race_id: str) -> Optional[Tuple[Athlete, Result, StartEntry]]:
    """Parse one `a.table-row` into an athlete plus its result/start entry."""
    cells = _row_cells(row)
    if not cells:
        return None

    href = row.get("href", "") or ""
    competitor = re.search(r"competitorid=(\d+)", href)

    rank: Optional[int] = None
    status = "OK"
    head = cells[0]
    if RE_INT.match(head):
        rank = int(head)
        cells = cells[1:]
    elif head.upper() in STATUS_TOKENS:
        status = head.upper()
        cells = cells[1:]

    fis_code = name = nation = None
    birth_year = bib = None
    times: List[float] = []
    diff_s = points = None

    for cell in cells:
        upper = cell.upper()
        if fis_code is None and RE_FIS_CODE.match(cell) and not RE_YEAR.match(cell):
            fis_code = cell
        elif birth_year is None and RE_YEAR.match(cell):
            birth_year = int(cell)
        elif nation is None and RE_NATION.match(cell):
            nation = cell
        elif bib is None and RE_INT.match(cell):
            bib = int(cell)
        elif cell.startswith("+") and (RE_CLOCK.match(cell) or RE_SECONDS.match(cell)):
            if diff_s is None:
                diff_s = _clock_to_seconds(cell)
        elif RE_CLOCK.match(cell):
            value = _clock_to_seconds(cell)
            if value is not None and value not in times:
                times.append(value)   # the same time appears twice per row
        elif RE_POINTS.match(cell):
            points = float(cell)
        elif upper in STATUS_TOKENS:
            status = upper
        elif name is None and re.search(r"[A-Za-z]", cell) and len(cell) > 3:
            name = cell

    if fis_code is None or name is None:
        return None
    if rank is None and status == "OK" and not times:
        status = "ENTERED"   # start list row: nothing has happened yet

    athlete = Athlete(fis_code=fis_code, name=name, nation=nation,
                      birth_year=birth_year,
                      competitor_id=competitor.group(1) if competitor else None)
    result = Result(race_id=race_id, fis_code=fis_code, rank=rank, bib=bib,
                    time_s=times[0] if times else None, diff_s=diff_s,
                    fis_points=points, status=status)
    start = StartEntry(race_id=race_id, fis_code=fis_code, bib=bib)
    return athlete, result, start


def parse_race_page(html: str, race_id: str, season: Optional[int] = None,
                    event_id: Optional[str] = None) -> RaceEntries:
    soup = _soup(html)
    race = parse_race_header(html, race_id, season=season, event_id=event_id)
    entries = RaceEntries(race=race)

    for row in soup.select("a.table-row"):
        parsed = parse_row(row, race_id)
        if parsed is None:
            continue
        athlete, result, start = parsed
        entries.athletes.append(athlete)
        entries.results.append(result)
        entries.start_list.append(start)

    # A page that has produced no ranks is a start list, and storing empty
    # result rows for it would poison the ratings.
    if not entries.has_results:
        entries.results = []
    return entries


# --------------------------------------------------------------------- crawling

def crawl_race(fetcher: Fetcher, race_id: str, season: Optional[int] = None,
               event_id: Optional[str] = None, force: bool = False) -> Optional[RaceEntries]:
    html = fetcher.get_optional(race_url(race_id), force=force)
    if html is None:
        log.warning("race %s: 404", race_id)
        return None
    return parse_race_page(html, race_id, season=season, event_id=event_id)


def crawl_season(fetcher: Fetcher, season: int, force: bool = False) -> Iterator[RaceEntries]:
    """Yield every World Cup race of a season, event by event."""
    calendar = fetcher.get(calendar_url(season), force=force)
    event_ids = parse_calendar(calendar)
    log.info("season %s: %d events", season, len(event_ids))

    for event_id in event_ids:
        page = fetcher.get(event_url(event_id, season), force=force)
        race_ids = parse_event(page)
        log.info("event %s: %d races", event_id, len(race_ids))
        for race_id in race_ids:
            entries = crawl_race(fetcher, race_id, season=season,
                                 event_id=event_id, force=force)
            if entries is not None:
                yield entries
