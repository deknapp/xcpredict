"""Turn ratings plus a start list into finishing-order probabilities.

The prediction is always conditioned on the *published start list*: only
athletes actually entered in the race can appear in the output. That is the
whole point of scraping start lists rather than predicting over a season-long
field.

Method: sample each starter's performance as ``theta_i + Gumbel(0, spread)``
where ``theta = rating * ln(10) / 400``, and rank by the sample. With
``spread = 1`` this is a Plackett-Luce model whose head-to-head probabilities
match the Elo ratings exactly, so the simulation and the ratings cannot drift
apart. Raising `spread` above 1 makes the race more of a lottery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence

import numpy as np

from .rating.elo import DEFAULT_RATING, SCALE, EloModel

#: rating points -> log-strength, so Gumbel sampling reproduces Elo odds
THETA_PER_POINT = math.log(10.0) / SCALE


@dataclass
class Starter:
    fis_code: str
    name: str
    nation: Optional[str] = None
    bib: Optional[int] = None
    rating: float = DEFAULT_RATING
    n_races: int = 0

    @property
    def unrated(self) -> bool:
        return self.n_races == 0


@dataclass
class AthletePrediction:
    fis_code: str
    name: str
    nation: Optional[str]
    rating: float
    n_races: int
    mean_rank: float
    median_rank: float
    p_win: float
    p_podium: float
    p_top10: float

    @property
    def unrated(self) -> bool:
        return self.n_races == 0


@dataclass
class RacePrediction:
    race_id: str
    pool: str
    n_starters: int
    n_sims: int
    athletes: List[AthletePrediction]
    unrated: List[str]

    def top(self, n: int = 15) -> List[AthletePrediction]:
        return self.athletes[:n]


def starters_from_db(conn, race_id: str, model: EloModel, pool: str,
                     when: Optional[date] = None) -> List[Starter]:
    from . import db

    rows = db.start_list(conn, race_id)
    starters = []
    for row in rows:
        entry = model.get(row["fis_code"], pool)
        starters.append(Starter(
            fis_code=row["fis_code"],
            name=row["name"] or row["fis_code"],
            nation=row["nation"],
            bib=row["bib"],
            rating=model.rating_on(row["fis_code"], pool, when),
            n_races=entry.n_races,
        ))
    return starters


def simulate(starters: Sequence[Starter], race_id: str = "", pool: str = "distance",
             n_sims: int = 20000, spread: float = 1.0,
             seed: Optional[int] = None) -> RacePrediction:
    if not starters:
        raise ValueError("no starters: scrape or supply a start list first")

    rng = np.random.default_rng(seed)
    theta = np.array([s.rating for s in starters]) * THETA_PER_POINT
    noise = rng.gumbel(loc=0.0, scale=spread, size=(n_sims, len(starters)))
    scores = theta[None, :] + noise

    # rank 1 = best score
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, len(starters) + 1)[None, :], axis=1)

    predictions = []
    for i, starter in enumerate(starters):
        column = ranks[:, i]
        predictions.append(AthletePrediction(
            fis_code=starter.fis_code,
            name=starter.name,
            nation=starter.nation,
            rating=starter.rating,
            n_races=starter.n_races,
            mean_rank=float(column.mean()),
            median_rank=float(np.median(column)),
            p_win=float((column == 1).mean()),
            p_podium=float((column <= 3).mean()),
            p_top10=float((column <= 10).mean()),
        ))

    predictions.sort(key=lambda p: p.mean_rank)
    return RacePrediction(
        race_id=race_id,
        pool=pool,
        n_starters=len(starters),
        n_sims=n_sims,
        athletes=predictions,
        unrated=[s.fis_code for s in starters if s.unrated],
    )


def format_table(prediction: RacePrediction, limit: int = 15) -> str:
    header = (f"{'#':>3}  {'Athlete':<26} {'Nat':<4} {'Elo':>6} {'Win':>7} "
              f"{'Podium':>7} {'Top10':>7} {'ERank':>6}")
    lines = [header, "-" * len(header)]
    for i, athlete in enumerate(prediction.top(limit), start=1):
        flag = "*" if athlete.unrated else " "
        lines.append(
            f"{i:>3}{flag} {athlete.name[:25]:<26} {athlete.nation or '':<4} "
            f"{athlete.rating:>6.0f} {athlete.p_win:>6.1%} {athlete.p_podium:>7.1%} "
            f"{athlete.p_top10:>7.1%} {athlete.mean_rank:>6.1f}"
        )
    if prediction.unrated:
        lines.append("")
        lines.append(f"* {len(prediction.unrated)} starter(s) have no prior World Cup "
                     f"result and are held at the default rating.")
    return "\n".join(lines)
