"""Elo ratings fitted to finishing order.

Why Elo and not a time model: World Cup formats are not comparable on the
clock. A mass start is tactical, a sprint is heats, and course and wax vary
enormously between venues, so finishing *order* is the only signal that means
the same thing everywhere. Order is exactly what Elo consumes.

Sprint and distance ability are rated in separate pools; they correlate, but
not enough to pool them.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

DEFAULT_RATING = 1500.0
# Elo's own scale: a 400-point edge is 10:1 odds in a head-to-head.
SCALE = 400.0


@dataclass
class EloConfig:
    k_base: float = 24.0
    #: Provisional athletes move faster until they have this many races.
    provisional_races: int = 10
    k_provisional: float = 60.0
    #: Ratings decay toward the mean while an athlete is not racing, so a
    #: skier who has been out for two seasons is not still rated on old form.
    half_life_days: float = 540.0
    default_rating: float = DEFAULT_RATING


@dataclass
class Rating:
    fis_code: str
    pool: str
    rating: float = DEFAULT_RATING
    n_races: int = 0
    last_race_id: Optional[str] = None
    last_date: Optional[date] = None


def expected_score(rating_a: float, rating_b: float) -> float:
    """P(A finishes ahead of B)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / SCALE))


def _parse_date(value) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


class EloModel:
    """Incremental multiplayer Elo. Feed races oldest first."""

    def __init__(self, config: Optional[EloConfig] = None):
        self.config = config or EloConfig()
        self.ratings: Dict[Tuple[str, str], Rating] = {}

    # ------------------------------------------------------------- accessors

    def get(self, fis_code: str, pool: str) -> Rating:
        key = (fis_code, pool)
        if key not in self.ratings:
            self.ratings[key] = Rating(fis_code=fis_code, pool=pool,
                                       rating=self.config.default_rating)
        return self.ratings[key]

    def rating_on(self, fis_code: str, pool: str, when: Optional[date] = None) -> float:
        """Rating for an athlete, decayed toward the mean up to `when`."""
        entry = self.get(fis_code, pool)
        return self._decayed(entry, when)

    def _decayed(self, entry: Rating, when: Optional[date]) -> float:
        if when is None or entry.last_date is None:
            return entry.rating
        days = (when - entry.last_date).days
        if days <= 0:
            return entry.rating
        weight = 0.5 ** (days / self.config.half_life_days)
        return (self.config.default_rating
                + weight * (entry.rating - self.config.default_rating))

    def _k(self, entry: Rating) -> float:
        if entry.n_races < self.config.provisional_races:
            return self.config.k_provisional
        return self.config.k_base

    # -------------------------------------------------------------- updating

    def update_race(self, finishers: Sequence[Tuple[str, int]], pool: str,
                    race_id: Optional[str] = None,
                    when: Optional[date] = None) -> None:
        """Apply one race.

        `finishers` is (fis_code, rank) for ranked athletes only. DNF/DNS are
        excluded by the caller: not finishing is usually equipment, illness or
        a crash, which says little about speed.
        """
        field = [(code, rank) for code, rank in finishers if rank is not None]
        n = len(field)
        if n < 3:
            return

        current = {code: self.rating_on(code, pool, when) for code, _ in field}
        deltas = {code: 0.0 for code in current}

        for i, (code_i, rank_i) in enumerate(field):
            for code_j, rank_j in field[i + 1:]:
                if rank_i == rank_j:
                    actual = 0.5
                else:
                    actual = 1.0 if rank_i < rank_j else 0.0
                expected = expected_score(current[code_i], current[code_j])
                surprise = actual - expected
                deltas[code_i] += surprise
                deltas[code_j] -= surprise

        for code, _rank in field:
            entry = self.get(code, pool)
            # Normalise by opponents faced so a 90-skier field does not move
            # ratings 3x as far as a 30-skier field.
            entry.rating = current[code] + self._k(entry) * deltas[code] / (n - 1)
            entry.n_races += 1
            entry.last_race_id = race_id
            entry.last_date = when

    # ------------------------------------------------------------------- I/O

    def to_rows(self) -> List[tuple]:
        return [(r.fis_code, r.pool, r.rating, r.n_races, r.last_race_id,
                 r.last_date.isoformat() if r.last_date else None)
                for r in self.ratings.values()]

    def load_rows(self, rows: Iterable) -> None:
        for row in rows:
            entry = Rating(fis_code=row["fis_code"], pool=row["pool"],
                           rating=row["rating"], n_races=row["n_races"],
                           last_race_id=row["last_race_id"],
                           last_date=_parse_date(row["last_date"]))
            self.ratings[(entry.fis_code, entry.pool)] = entry


def pool_for(race) -> str:
    """Rating pool a race belongs to. `race` is a Race or a sqlite Row."""
    kind = race["kind"] if hasattr(race, "keys") else race.kind
    return kind or "distance"


def fit(conn, config: Optional[EloConfig] = None,
        include_team: bool = False) -> EloModel:
    """Fit ratings over every finished race in the database, oldest first."""
    from .. import db

    model = EloModel(config)
    races = db.finished_races(conn, include_team=include_team)
    for race in races:
        results = db.race_results(conn, race["race_id"])
        model.update_race(
            [(r["fis_code"], r["rank"]) for r in results
             if (r["status"] or "OK") == "OK"],
            pool=pool_for(race),
            race_id=race["race_id"],
            when=_parse_date(race["race_date"]),
        )
    log.info("fitted %d ratings over %d races", len(model.ratings), len(races))
    return model


def save(conn, model: EloModel) -> None:
    conn.execute("DELETE FROM ratings")
    conn.executemany(
        """INSERT INTO ratings (fis_code, pool, rating, n_races, last_race_id, last_date)
           VALUES (?, ?, ?, ?, ?, ?)""",
        model.to_rows(),
    )
    conn.commit()


def load(conn, config: Optional[EloConfig] = None) -> EloModel:
    model = EloModel(config)
    model.load_rows(conn.execute("SELECT * FROM ratings"))
    return model
