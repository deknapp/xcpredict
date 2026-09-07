"""What a skier has done in races *like this one*.

Elo gives each athlete a single number. That is the model's central weakness
and this module is the answer to it: a sprinter and a 50 km specialist are not
the same athlete, and pooling their results into one rating throws away most of
what the results actually say.

The idea here is the one a coach would use. To predict Diggins in a 10 km
skate, look hardest at her past 10 km skates, then at her 15 km skates, then at
her 10 km classics, and barely at her sprints. Nothing is discarded -- a sprint
result is weak evidence about a 10 km, not zero evidence -- but the weights
differ, and they differ smoothly.

So every historical result gets a weight against the target race:

    w = technique_similarity x kind_similarity x length_similarity x recency

and a skier's features are weighted summaries of their performance under that
weighting. The weights are a *kernel*, not a rule: a 2 km prologue and a sprint
qualifier end up close together because their lengths are close, without anyone
hard-coding "treat prologues as sprints".

**Causality.** Every feature for a race is built only from races strictly
before it. This is the thing that is easy to get wrong and fatal when you do:
ratings fitted on the whole season leak the answer into a prediction of a race
inside that season, the model looks superb, and none of it survives contact
with a race that has not happened yet.

**Performance measure.** Finishing position is converted to a percentile within
its field, so a 10th of 90 counts as a better ride than 10th of 12. Time behind
the winner would be better still and is available on only 61% of results, so
rank percentile is what everything is built on and time is left for later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence

from .rating.elo import _parse_date

# ---------------------------------------------------------------- similarity

#: Half-life in days for how fast old results stop mattering. Roughly a season
#: and a half: last winter is strong evidence, three winters ago is weak.
RECENCY_HALF_LIFE_DAYS = 400.0

#: How sharply length similarity falls off, in natural-log units of distance
#: ratio. At 0.5, a 10 km judged against a 15 km keeps about 45% weight and
#: against a 1.5 km sprint about 2%. Log-ratio rather than absolute difference
#: because 5 km versus 10 km is a bigger jump in kind than 45 km versus 50 km.
LENGTH_SIGMA_LOG = 0.5

#: Weight retained when the technique differs. Not zero: a strong classic skier
#: is usually a decent skate skier, and treating the two as unrelated discards
#: half the record for anyone who races both.
TECHNIQUE_MISMATCH = 0.45

#: Skiathlon is half classic and half free, so it is partially similar to both
#: rather than similar to neither.
TECHNIQUE_COMBINED = 0.75

#: Weight retained across the sprint/distance divide, before length similarity
#: is also applied. Deliberately small but non-zero.
KIND_MISMATCH = 0.30

#: Assumed length when a race has none recorded, by kind. Only used so that a
#: missing value degrades to something sensible instead of dropping the result.
ASSUMED_LENGTH_KM = {"sprint": 1.5, "distance": 12.0}


def _length_of(race) -> float:
    length = race["length_km"] if "length_km" in race.keys() else None
    if length and length > 0:
        return float(length)
    return ASSUMED_LENGTH_KM.get(race["kind"], 8.0)


def technique_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if "CF" in (a, b):
        return TECHNIQUE_COMBINED
    if "?" in (a, b):
        # Unknown technique should not be treated as a mismatch; it is missing
        # information, and penalising it would silently down-weight whole
        # seasons that happened to be scraped without it.
        return 1.0
    return TECHNIQUE_MISMATCH


def length_similarity(target_km: float, other_km: float) -> float:
    if target_km <= 0 or other_km <= 0:
        return 1.0
    ratio = math.log(other_km / target_km)
    return math.exp(-(ratio * ratio) / (2 * LENGTH_SIGMA_LOG * LENGTH_SIGMA_LOG))


def kind_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if "?" in (a, b):
        return 1.0
    return KIND_MISMATCH


def recency_weight(then: Optional[date], now: Optional[date]) -> float:
    if then is None or now is None:
        return 1.0
    days = (now - then).days
    if days < 0:
        # A result from after the target race must never inform it.
        return 0.0
    return 0.5 ** (days / RECENCY_HALF_LIFE_DAYS)


def race_similarity(target, other) -> float:
    """How much ``other`` should count when predicting ``target``."""
    return (
        technique_similarity(target["technique"], other["technique"])
        * kind_similarity(target["kind"], other["kind"])
        * length_similarity(_length_of(target), _length_of(other))
    )


# ------------------------------------------------------------------ history


@dataclass
class Performance:
    """One past result, reduced to what the model needs."""

    race_date: Optional[date]
    kind: str
    technique: str
    length_km: float
    percentile: float          # 0 = won, 1 = last
    field_size: int
    fis_points: Optional[float]
    finished: bool


#: Names of the features produced, in a fixed order. Kept explicit because the
#: model's learned weights are only interpretable next to the names.
FEATURE_NAMES = [
    "similar_form",        # weighted mean percentile in similar races (inverted)
    "similar_best",        # weighted best percentile (inverted)
    "overall_form",        # weighted mean percentile across everything
    "specificity",         # similar_form minus overall_form: is this their event?
    "experience",          # log of effective number of similar races
    "recent_starts",       # log count of starts in the last 90 days
    "fis_points",          # most recent *prior* FIS points, scaled and inverted
    "has_fis_points",      # whether the above is real or a fallback
    "consistency",         # 1 - weighted stdev of percentile
]


def performances_before(
    history: Sequence[Performance], cutoff: Optional[date]
) -> List[Performance]:
    if cutoff is None:
        return list(history)
    return [p for p in history if p.race_date is not None and p.race_date < cutoff]


def _weighted_stats(values: List[float], weights: List[float]):
    total = sum(weights)
    if total <= 0:
        return None, None
    mean = sum(v * w for v, w in zip(values, weights)) / total
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total
    return mean, math.sqrt(max(0.0, variance))


def build_features(
    history: Sequence[Performance],
    target_race,
    target_date: Optional[date],
) -> Optional[List[float]]:
    """Feature vector for one athlete in one race, or None if unrateable.

    Returns None when the athlete has no prior results at all. A debutant
    cannot be predicted from their record because they do not have one, and
    inventing an average for them would quietly fill the field with fictional
    mid-pack skiers.
    """
    past = performances_before(history, target_date)
    if not past:
        return None

    target_len = _length_of(target_race)
    weights, percentiles = [], []
    for p in past:
        similarity = (
            technique_similarity(target_race["technique"], p.technique)
            * kind_similarity(target_race["kind"], p.kind)
            * length_similarity(target_len, p.length_km)
        )
        weight = similarity * recency_weight(p.race_date, target_date)
        if weight <= 0:
            continue
        weights.append(weight)
        percentiles.append(p.percentile)

    if not weights:
        return None

    similar_mean, similar_sd = _weighted_stats(percentiles, weights)

    # Overall form uses recency only, so "specificity" below is a genuine
    # comparison of this event against the athlete's general level rather than
    # two differently-weighted versions of the same thing.
    recency_only = [recency_weight(p.race_date, target_date) for p in past]
    overall_mean, _ = _weighted_stats([p.percentile for p in past], recency_only)
    overall_mean = overall_mean if overall_mean is not None else similar_mean

    # Best result, taken as the weighted 20th percentile rather than the single
    # minimum: one lucky day should not define an athlete.
    ordered = sorted(zip(percentiles, weights))
    cumulative, cutoff_weight, best = 0.0, 0.2 * sum(weights), ordered[0][0]
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= cutoff_weight:
            best = value
            break

    effective_n = sum(weights)
    recent = [p for p in past if p.race_date and target_date
              and (target_date - p.race_date).days <= 90]

    # A finish-rate feature was here and was removed: the scrape records no
    # DNFs for individual races (every non-finisher in the database is a relay
    # start-list entry), so it was a constant 1.0 for every athlete -- a column
    # of no information that the fit still had to spend a weight on. Worth
    # restoring if the scraper ever captures DNFs.
    points = next((p.fis_points for p in reversed(past)
                   if p.fis_points is not None and p.fis_points > 0), None)

    return [
        1.0 - similar_mean,                       # higher is better
        1.0 - best,
        1.0 - overall_mean,
        overall_mean - similar_mean,              # positive => better here than usual
        math.log1p(effective_n),
        math.log1p(len(recent)),
        1.0 - min(points, 300.0) / 300.0 if points is not None else 0.5,
        1.0 if points is not None else 0.0,
        1.0 - min(1.0, similar_sd if similar_sd is not None else 0.5),
    ]


def histories_from_results(rows: Iterable[dict]) -> Dict[str, List[Performance]]:
    """Group result rows into per-athlete history, ordered by date.

    ``rows`` must carry the race columns joined on: kind, technique,
    length_km, race_date, plus rank, field size and status.
    """
    histories: Dict[str, List[Performance]] = {}
    for row in rows:
        code = row["fis_code"]
        histories.setdefault(code, []).append(
            Performance(
                race_date=_parse_date(row["race_date"]),
                kind=row["kind"],
                technique=row["technique"],
                length_km=float(row["length_km"]) if row["length_km"] else
                ASSUMED_LENGTH_KM.get(row["kind"], 8.0),
                percentile=row["percentile"],
                field_size=row["field_size"],
                fis_points=row["fis_points"],
                finished=row["rank"] is not None,
            )
        )
    for entries in histories.values():
        entries.sort(key=lambda p: p.race_date or date.min)
    return histories
