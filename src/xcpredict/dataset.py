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

from . import strength as strength_mod
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


#: Substrings marking a race as World Cup. Matched on the series name FIS
#: prints on the page rather than on a category code, because the code is not
#: stored per race and the name is.
WORLD_CUP_MARKERS = ("world cup",)

#: Series that are never useful regardless of who started: youth and children's
#: racing, where a stray World Cup athlete's presence would be a data error
#: rather than a signal.
EXCLUDED_MARKERS = ("children", "youth", "u16", "u14")


def is_world_cup(race) -> bool:
    series = (race.get("series") or "").lower()
    return any(marker in series for marker in WORLD_CUP_MARKERS)


def is_excluded(race) -> bool:
    series = (race.get("series") or "").lower()
    return any(marker in series for marker in EXCLUDED_MARKERS)


def load_races(conn, *, include_team: bool = False) -> List[dict]:
    sql = """
        SELECT race_id, season, race_date, place, title, gender, series,
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
    use_other_series: bool = True,
    world_cup_only_samples: bool = True,
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
    wc_starts: Dict[str, int] = {}
    skipped_weak = skipped_excluded = 0

    for race in races:
        world_cup = is_world_cup(race)

        if is_excluded(race):
            skipped_excluded += 1
            continue
        if not world_cup and not use_other_series:
            continue
        race_date = _parse_date(race["race_date"])
        results = load_results(conn, race["race_id"])
        if not results:
            continue

        finishers = [r for r in results if r["rank"] is not None]
        field_size = len(results)

        # A race outside the World Cup only counts if enough of its field are
        # skiers we already know. Without that anchoring the result is not
        # comparable with anything else in the data, however fast it was.
        if world_cup:
            field = strength_mod.FieldStrength(
                n_anchors=len(finishers), n_starters=field_size,
                anchor_quality=1.0, strength=1.0, usable=True)
        else:
            anchors = strength_mod.count_anchors(
                [r["fis_code"] for r in results], wc_starts)
            standings = []
            for code in anchors:
                past = history.get(code, [])
                if past:
                    recent = past[-12:]
                    standings.append(
                        1.0 - sum(p.percentile for p in recent) / len(recent))
            field = strength_mod.assess(standings, field_size)
            if not field.usable:
                skipped_weak += 1
                continue

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

        wanted = world_cup or not world_cup_only_samples
        if wanted and len(rows) >= min_rateable:
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

        # Only now does this race enter the record. The percentile stored is
        # adjusted for how strong the field was, so a win against nobody does
        # not read like a win against the world. Ordering inside the race is
        # untouched -- the adjustment rescales, it never reorders.
        length = race["length_km"] or 0.0
        for entry in results:
            raw = _percentile(entry["rank"], field_size)
            history.setdefault(entry["fis_code"], []).append(Performance(
                race_date=race_date,
                kind=race["kind"],
                technique=race["technique"],
                length_km=float(length) if length else
                (1.5 if race["kind"] == "sprint" else 12.0),
                percentile=strength_mod.adjust_percentile(raw, field.strength),
                field_size=field_size,
                fis_points=entry["fis_points"],
                finished=entry["rank"] is not None,
            ))

        if world_cup:
            for entry in results:
                wc_starts[entry["fis_code"]] = wc_starts.get(entry["fis_code"], 0) + 1

    log.info(
        "built %d sample races from %d total (%d skipped as too weak, "
        "%d excluded series)",
        len(samples), len(races), skipped_weak, skipped_excluded,
    )
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
