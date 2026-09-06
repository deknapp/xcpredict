"""Walk-forward backtest.

Races are replayed in chronological order. Each race is predicted using only
ratings fitted on races *before* it, then the model is updated with the real
result. The field used for a past race is its actual finishers, which is what
a published start list gives you for a future race — so the backtest measures
the same task the predictor performs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .rating.elo import EloConfig, EloModel, _parse_date, expected_score, pool_for


@dataclass
class Backtest:
    n_races: int = 0
    n_pairs: int = 0
    pair_correct: float = 0.0
    log_loss_sum: float = 0.0
    winner_ranks: List[int] = field(default_factory=list)
    spearman_sum: float = 0.0
    #: races skipped because too few starters had a prior rating
    n_skipped: int = 0

    @property
    def pair_accuracy(self) -> float:
        return self.pair_correct / self.n_pairs if self.n_pairs else float("nan")

    @property
    def log_loss(self) -> float:
        return self.log_loss_sum / self.n_pairs if self.n_pairs else float("nan")

    @property
    def mean_spearman(self) -> float:
        return self.spearman_sum / self.n_races if self.n_races else float("nan")

    @property
    def winner_top3_rate(self) -> float:
        if not self.winner_ranks:
            return float("nan")
        return sum(1 for r in self.winner_ranks if r <= 3) / len(self.winner_ranks)

    def summary(self) -> str:
        return "\n".join([
            f"races scored          {self.n_races}  (skipped {self.n_skipped})",
            f"pairwise accuracy     {self.pair_accuracy:.3f}",
            f"pairwise log loss     {self.log_loss:.4f}",
            f"mean rank correlation {self.mean_spearman:.3f}",
            f"winner in pred top-3  {self.winner_top3_rate:.3f}",
        ])


def _spearman(predicted: List[float], actual: List[float]) -> float:
    n = len(predicted)
    if n < 2:
        return float("nan")
    mean_p = sum(predicted) / n
    mean_a = sum(actual) / n
    num = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
    den = math.sqrt(sum((p - mean_p) ** 2 for p in predicted)
                    * sum((a - mean_a) ** 2 for a in actual))
    return num / den if den else float("nan")


def run(conn, config: Optional[EloConfig] = None, min_rated: int = 10,
        include_team: bool = False) -> Backtest:
    from . import db

    model = EloModel(config)
    report = Backtest()

    for race in db.finished_races(conn, include_team=include_team):
        pool = pool_for(race)
        when = _parse_date(race["race_date"])
        results = [r for r in db.race_results(conn, race["race_id"])
                   if (r["status"] or "OK") == "OK"]
        if len(results) < 3:
            continue

        rated = [r for r in results
                 if model.get(r["fis_code"], pool).n_races > 0]
        if len(rated) >= min_rated:
            _score_race(model, report, rated, pool, when)
        else:
            report.n_skipped += 1

        model.update_race([(r["fis_code"], r["rank"]) for r in results],
                          pool=pool, race_id=race["race_id"], when=when)

    return report


def _score_race(model: EloModel, report: Backtest, rated, pool, when) -> None:
    ratings = [model.rating_on(r["fis_code"], pool, when) for r in rated]
    actual = [float(r["rank"]) for r in rated]

    for i in range(len(rated)):
        for j in range(i + 1, len(rated)):
            probability = expected_score(ratings[i], ratings[j])
            beat = actual[i] < actual[j]
            report.n_pairs += 1
            report.pair_correct += 1.0 if (probability > 0.5) == beat else 0.0
            p = probability if beat else 1.0 - probability
            report.log_loss_sum -= math.log(max(p, 1e-12))

    # higher rating should mean lower (better) finishing rank
    report.spearman_sum += _spearman([-r for r in ratings], actual)
    report.n_races += 1

    best = min(range(len(rated)), key=lambda i: actual[i])
    predicted_rank = sorted(range(len(rated)), key=lambda i: -ratings[i]).index(best) + 1
    report.winner_ranks.append(predicted_rank)
