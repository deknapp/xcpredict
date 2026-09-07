"""Turning the database into training rows, without letting the future in.

The single thing this module exists to get right: a race's features must be
built only from races that finished before it. Everything else here is
plumbing.

That constraint is easy to state and easy to violate, because the natural way
to write it -- load all results, group by athlete, compute their averages -- is
wrong in a way that produces excellent numbers. An average that includes the
race being predicted knows the answer. The model then looks superb in backtest,
and fails completely on a race that has not happened, which is the only kind
anyone cares about.

So the loop here is strictly chronological: walk races in date order, build
features from the history accumulated *so far*, then append that race's results
to the history. A race is never in its own feature set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import Performance, build_features
from .rating.elo import _parse_date

log = logging.getLogger(__name__)

#: A race needs at least this many rateable starters to be worth including.
#: Below it the pairwise signal is thin and the field is usually a fragment of
#: a result page rather than a real race.
MIN_RATEABLE = 6


@dataclass
class RaceSample:
    """One race, ready to train or evaluate on."""

    race_id: str
    race_date: Optional[date]
    season: Optional[int]
    place: Optional[str]
    title: Optional[str]
    gender: Optional[str]
    kind: str
    technique: str
    length_km: Optional[float]
    fis_codes: List[str]           # in finishing order
    features: np.ndarray           # one row per athlete, same order
    fis_points: List[Optional[float]]   # for the baseline

    @property
    def n(self) -> int:
        return len(self.fis_codes)


def _percentile(rank: Optional[int], field_size: int) -> float:
    """0 for the winner, 1 for last. DNF counts as worse than last finisher."""
    if rank is None:
        return 1.0
    if field_size <= 1:
        return 0.0
    return (rank - 1) / (field_size - 1)


def load_races(conn, *, include_team: bool = False) -> List[dict]:
    sql = """
        SELECT race_id, season, race_date, place, title, gender,
               kind, technique, start_type, length_km, is_team
        FROM races
        WHERE race_date IS NOT NULL
    """
    if not include_team:
        sql += " AND (is_team IS NULL OR is_team = 0)"
    sql += " ORDER BY race_date, race_id"
    return [dict(r) for r in conn.execute(sql)]


def load_results(conn, race_id: str) -> List[dict]:
    rows = [dict(r) for r in conn.execute(
        """SELECT fis_code, rank, time_s, fis_points, status
           FROM results WHERE race_id = ?""", (race_id,))]
    # Finishers in order first, then non-finishers. Sorting None last matters:
    # a DNF is not a good result and must not sort to the front.
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0))
    return rows


def build_samples(
    conn,
    *,
    min_rateable: int = MIN_RATEABLE,
    include_team: bool = False,
) -> List[RaceSample]:
    """Every race, with causally-safe features. Chronological order.

    Only finishers become training rows. A DNF still updates the athlete's
    history -- it is evidence about reliability, and ``finish_rate`` uses it --
    but it has no finishing position, so it cannot take part in a pairwise
    comparison about who beat whom.
    """
    races = load_races(conn, include_team=include_team)
    history: Dict[str, List[Performance]] = {}
    samples: List[RaceSample] = []

    for race in races:
        race_date = _parse_date(race["race_date"])
        results = load_results(conn, race["race_id"])
        if not results:
            continue

        finishers = [r for r in results if r["rank"] is not None]
        field_size = len(results)

        rows, codes, points = [], [], []
        for entry in finishers:
            feats = build_features(
                history.get(entry["fis_code"], []), race, race_date
            )
            if feats is None:
                continue          # no prior record; cannot be predicted
            rows.append(feats)
            codes.append(entry["fis_code"])
            points.append(entry["fis_points"])

        if len(rows) >= min_rateable:
            samples.append(RaceSample(
                race_id=race["race_id"],
                race_date=race_date,
                season=race["season"],
                place=race["place"],
                title=race["title"],
                gender=race["gender"],
                kind=race["kind"],
                technique=race["technique"],
                length_km=race["length_km"],
                fis_codes=codes,
                features=np.array(rows, dtype=float),
                fis_points=points,
            ))

        # Only now does this race enter the record.
        length = race["length_km"] or 0.0
        for entry in results:
            history.setdefault(entry["fis_code"], []).append(Performance(
                race_date=race_date,
                kind=race["kind"],
                technique=race["technique"],
                length_km=float(length) if length else
                (1.5 if race["kind"] == "sprint" else 12.0),
                percentile=_percentile(entry["rank"], field_size),
                field_size=field_size,
                fis_points=entry["fis_points"],
                finished=entry["rank"] is not None,
            ))

    log.info("built %d rateable races from %d total", len(samples), len(races))
    return samples


def split_by_season(
    samples: Sequence[RaceSample], holdout_seasons: Sequence[int]
) -> Tuple[List[RaceSample], List[RaceSample]]:
    """Train on earlier seasons, test on later ones.

    Held out by season rather than at random. A random split would put races
    from the same weekend on both sides, and since form persists across a
    weekend that leaks the answer -- the model would be tested on athletes
    whose current condition it had already seen.
    """
    holdout = set(holdout_seasons)
    train = [s for s in samples if s.season not in holdout]
    test = [s for s in samples if s.season in holdout]
    return train, test


def as_arrays(samples: Sequence[RaceSample]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The shape :func:`xcpredict.ml.fit` wants."""
    return [(s.features, np.arange(s.n)) for s in samples]
