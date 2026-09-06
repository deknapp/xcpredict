"""SQLite persistence layer."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Athlete, Race, RaceEntries, Result, StartEntry

DEFAULT_DB = Path("data/xcpredict.sqlite")
_SCHEMA = Path(__file__).with_name("schema.sql")


def connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA.read_text())
    return conn


def _iso(value) -> Optional[str]:
    return value.isoformat() if value is not None else None


def upsert_athletes(conn: sqlite3.Connection, athletes: Iterable[Athlete]) -> None:
    conn.executemany(
        """INSERT INTO athletes (fis_code, name, nation, birth_year, competitor_id)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(fis_code) DO UPDATE SET
               name=excluded.name,
               nation=COALESCE(excluded.nation, athletes.nation),
               birth_year=COALESCE(excluded.birth_year, athletes.birth_year),
               competitor_id=COALESCE(excluded.competitor_id, athletes.competitor_id)""",
        [(a.fis_code, a.name, a.nation, a.birth_year, a.competitor_id) for a in athletes],
    )


def upsert_race(conn: sqlite3.Connection, race: Race) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO races
           (race_id, event_id, season, race_date, place, series, title, gender,
            kind, technique, start_type, length_km, is_team)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (race.race_id, race.event_id, race.season, _iso(race.race_date), race.place,
         race.series, race.title, race.gender, race.kind, race.technique,
         race.start_type, race.length_km, int(race.is_team)),
    )


def upsert_results(conn: sqlite3.Connection, results: Iterable[Result]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO results
           (race_id, fis_code, rank, bib, time_s, diff_s, fis_points, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(r.race_id, r.fis_code, r.rank, r.bib, r.time_s, r.diff_s, r.fis_points, r.status)
         for r in results],
    )


def replace_start_list(conn: sqlite3.Connection, race_id: str,
                       entries: Iterable[StartEntry]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute("DELETE FROM start_list WHERE race_id = ?", (race_id,))
    conn.executemany(
        """INSERT INTO start_list (race_id, fis_code, bib, start_time, fetched_at)
           VALUES (?, ?, ?, ?, ?)""",
        [(e.race_id, e.fis_code, e.bib, e.start_time, now) for e in entries],
    )


def save_entries(conn: sqlite3.Connection, entries: RaceEntries) -> None:
    """Persist everything a single race page produced."""
    upsert_race(conn, entries.race)
    upsert_athletes(conn, entries.athletes)
    if entries.results:
        upsert_results(conn, entries.results)
    if entries.start_list:
        replace_start_list(conn, entries.race.race_id, entries.start_list)
    conn.commit()


def get_race(conn: sqlite3.Connection, race_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM races WHERE race_id = ?", (race_id,)).fetchone()


def start_list(conn: sqlite3.Connection, race_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT s.fis_code, s.bib, a.name, a.nation
           FROM start_list s LEFT JOIN athletes a USING (fis_code)
           WHERE s.race_id = ? ORDER BY s.bib""",
        (race_id,),
    ).fetchall()


def finished_races(conn: sqlite3.Connection, include_team: bool = False) -> List[sqlite3.Row]:
    """Races that have at least one ranked result, oldest first."""
    return conn.execute(
        """SELECT r.* FROM races r
           WHERE EXISTS (SELECT 1 FROM results x
                         WHERE x.race_id = r.race_id AND x.rank IS NOT NULL)
             AND (? OR r.is_team = 0)
           ORDER BY COALESCE(r.race_date, ''), r.race_id""",
        (1 if include_team else 0,),
    ).fetchall()


def race_results(conn: sqlite3.Connection, race_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM results
           WHERE race_id = ? AND rank IS NOT NULL
           ORDER BY rank""",
        (race_id,),
    ).fetchall()


def athlete_names(conn: sqlite3.Connection) -> dict:
    return {r["fis_code"]: (r["name"], r["nation"])
            for r in conn.execute("SELECT fis_code, name, nation FROM athletes")}
