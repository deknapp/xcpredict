"""What the model has to beat.

A pairwise accuracy of 0.800 sounds good and means nothing on its own. In a
sport where the favourites usually win, a rule as crude as "whoever has better
FIS points wins" already scores well, and any model that does not clearly beat
it has earned nothing -- FIS publishes those points for free.

So every result this project reports carries its floor and its competitors:

* **random** -- 0.5 by construction. The floor.
* **fis_points** -- order by the athlete's most recent FIS points. This is the
  real competitor: a published, free, official number.
* **recent_form** -- order by the athlete's recency-weighted mean finishing
  percentile, ignoring technique and distance entirely. This is the ablation
  that isolates the thing the model is *for*: if the similarity kernel adds
  nothing, this baseline matches it, and the kernel is decoration.
* **elo** -- the project's previous model, where ratings are available.

The third one is the sharpest test and the reason it exists. It is not enough
to beat FIS points; a model whose whole premise is that discipline and distance
matter must beat a version of itself that ignores them.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .dataset import RaceSample
from .features import FEATURE_NAMES

#: Index of the feature holding recency-weighted overall form, used by the
#: ablation baseline. Looked up by name so reordering features cannot silently
#: repoint it at something else.
OVERALL_FORM = FEATURE_NAMES.index("overall_form")
SIMILAR_FORM = FEATURE_NAMES.index("similar_form")
FIS_POINTS = FEATURE_NAMES.index("fis_points")


def _score_pairs(scores: Sequence[float]) -> tuple:
    """Correct pairs and total, given scores in true finishing order."""
    correct, total = 0.0, 0
    n = len(scores)
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if scores[i] > scores[j]:
                correct += 1
            elif scores[i] == scores[j]:
                correct += 0.5
    return correct, total


def evaluate_scorer(
    samples: Sequence[RaceSample],
    scorer: Callable[[RaceSample], Optional[Sequence[float]]],
) -> Dict[str, float]:
    """Pairwise accuracy of any scoring function over the sample races.

    ``scorer`` returns one score per athlete, higher meaning predicted to
    finish ahead, or None if it cannot score this race. Races it declines are
    counted, because a baseline that only answers the easy races is not
    comparable with one that answers all of them.
    """
    correct, total, races, skipped = 0.0, 0, 0, 0
    for sample in samples:
        scores = scorer(sample)
        if scores is None:
            skipped += 1
            continue
        c, t = _score_pairs(scores)
        correct += c
        total += t
        races += 1
    if not total:
        return {"races": 0, "skipped": skipped, "pairs": 0}
    return {
        "races": races,
        "skipped": skipped,
        "pairs": total,
        "pair_accuracy": round(correct / total, 4),
    }


# ----------------------------------------------------------------- scorers


def random_scorer(seed: int = 0) -> Callable[[RaceSample], Sequence[float]]:
    rng = np.random.default_rng(seed)
    return lambda sample: rng.random(sample.n)


def fis_points_scorer(sample: RaceSample) -> Optional[Sequence[float]]:
    """Order by each athlete's most recent FIS points *before* this race.

    The obvious version of this baseline is a trap, and the first version here
    fell in it. ``results.fis_points`` is the points **earned in that race**:
    the winner scores 0.0 and the value rises monotonically with finishing
    position. Ordering by it scored 0.9651 and was simply reading the result.

    The legitimate version uses the points the athlete carried in from earlier
    races, which is what a person consulting the FIS list before the start
    would have. That value is already computed causally as a model feature, so
    it is read from there rather than recomputed and risking the same mistake
    twice.
    """
    return sample.features[:, FIS_POINTS]


def recent_form_scorer(sample: RaceSample) -> Sequence[float]:
    """Form with no regard for technique, distance or discipline.

    The ablation. This is the model's own ``overall_form`` feature used alone,
    so the comparison isolates exactly one thing: whether weighting history by
    similarity to the target race adds anything over just knowing who has been
    going well lately.
    """
    return sample.features[:, OVERALL_FORM]


def similar_form_scorer(sample: RaceSample) -> Sequence[float]:
    """The similarity-weighted feature alone, with no learning at all.

    Sits between ``recent_form`` and the fitted model, and separates two
    questions that are easy to conflate: does the kernel help, and does
    *fitting weights over the features* help beyond the kernel?
    """
    return sample.features[:, SIMILAR_FORM]


def elo_scorer(ratings_by_race: Dict[str, Dict[str, float]]) -> Callable:
    """Elo ratings as of before each race, if a rating dump is available."""
    def score(sample: RaceSample) -> Optional[Sequence[float]]:
        table = ratings_by_race.get(sample.race_id)
        if not table:
            return None
        known = [table[c] for c in sample.fis_codes if c in table]
        if len(known) < 2:
            return None
        fallback = sum(known) / len(known)
        return [table.get(c, fallback) for c in sample.fis_codes]
    return score


def compare(
    samples: Sequence[RaceSample],
    model_scorer: Optional[Callable] = None,
    elo_ratings: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Every baseline plus the model, measured identically on the same races."""
    table = {
        "random": evaluate_scorer(samples, random_scorer()),
        "fis_points": evaluate_scorer(samples, fis_points_scorer),
        "recent_form (no kernel)": evaluate_scorer(samples, recent_form_scorer),
        "similar_form (kernel, unfitted)": evaluate_scorer(samples, similar_form_scorer),
    }
    if elo_ratings:
        table["elo"] = evaluate_scorer(samples, elo_scorer(elo_ratings))
    if model_scorer is not None:
        table["learned model"] = evaluate_scorer(samples, model_scorer)
    return table


def format_comparison(table: Dict[str, Dict[str, float]]) -> str:
    """A table where the number is next to what it must beat."""
    lines = [f"{'method':34s} {'races':>6s} {'pairs':>9s} {'pair acc':>9s}"]
    lines.append("-" * 62)
    ordered = sorted(table.items(),
                     key=lambda kv: kv[1].get("pair_accuracy", 0.0))
    for name, stats in ordered:
        if not stats.get("pairs"):
            lines.append(f"{name:34s} {'-':>6s} {'-':>9s} {'no data':>9s}")
            continue
        lines.append(
            f"{name:34s} {stats['races']:6d} {stats['pairs']:9,d} "
            f"{stats['pair_accuracy']:9.4f}"
        )
    return "\n".join(lines)
