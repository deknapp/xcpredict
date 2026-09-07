"""A learned ranker, fitted on every pair of skiers who have raced each other.

Elo has no features. It cannot know that a race is a classic sprint, that this
skier only ever does well in classic sprints, or that they last raced in
January. This model can, because its input is
:mod:`xcpredict.features` -- what each athlete has done in races *like this
one*, under a similarity kernel over technique, discipline, distance and time.

**Why pairwise.** The thing we are predicting is an order, not a number. So
the training examples are pairs: within a race, for every two finishers, did A
beat B? A pair is labelled 1 if it did. The model scores each athlete and is
fitted so that the difference in scores predicts the pair. This is the standard
RankNet objective, and it is exactly what the evaluation measures, which is
worth insisting on -- a model fitted to predict finishing *time* and then
scored on ordering is being graded on something it was not asked to do.

**Why linear, and why that is not a cop-out.** The score is a weighted sum of
the features, fitted by gradient descent. That buys three things that matter
more here than a fraction of a point of accuracy:

* The whole model is ~10 numbers, so it commits to git as a readable file and
  runs in a browser with no dependencies.
* The weights are interpretable. If "specificity" earns a large weight, that is
  a finding about the sport -- event specialisation predicts results -- and not
  merely a better score.
* With 934 athletes over four seasons, a boosted forest has ample room to
  memorise who is fast, which is what Elo already tells us. The interesting
  question is what *else* predicts the result, and a small model is a harsher
  test of that.

The non-linearity that matters lives in the features, where the similarity
kernel is already exponential in log-distance.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .features import FEATURE_NAMES

#: L2 penalty. Small but non-zero: several features are strongly correlated
#: (similar_form and overall_form especially), and without it their weights
#: drift apart in equal and opposite directions and stop meaning anything.
DEFAULT_L2 = 1e-4

DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.50   # converged by here; 3,000 epochs changes nothing

#: Pairs are sampled per race rather than enumerated. A 90-strong field has
#: 4,005 pairs and would swamp a 12-strong one; capping keeps races weighted
#: roughly evenly and keeps the fit quick.
MAX_PAIRS_PER_RACE = 600


@dataclass
class RankerModel:
    """Learned weights, plus everything needed to reproduce them."""

    weights: List[float]
    bias: float = 0.0
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    feature_mean: List[float] = field(default_factory=list)
    feature_std: List[float] = field(default_factory=list)
    trained_on_races: int = 0
    trained_on_pairs: int = 0
    notes: str = ""

    def score(self, features: Sequence[float]) -> float:
        x = np.asarray(features, dtype=float)
        if self.feature_mean:
            x = (x - np.asarray(self.feature_mean)) / np.asarray(self.feature_std)
        return float(np.dot(x, np.asarray(self.weights)) + self.bias)

    def score_many(self, matrix: np.ndarray) -> np.ndarray:
        x = matrix.astype(float)
        if self.feature_mean:
            x = (x - np.asarray(self.feature_mean)) / np.asarray(self.feature_std)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return x @ np.asarray(self.weights) + self.bias

    # ------------------------------------------------------------ storage

    def to_dict(self) -> dict:
        return {
            "kind": "pairwise_logistic_ranker",
            "version": 1,
            "feature_names": self.feature_names,
            "weights": [round(w, 6) for w in self.weights],
            "bias": round(self.bias, 6),
            "feature_mean": [round(v, 6) for v in self.feature_mean],
            "feature_std": [round(v, 6) for v in self.feature_std],
            "trained_on_races": self.trained_on_races,
            "trained_on_pairs": self.trained_on_pairs,
            "notes": self.notes,
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> RankerModel:
        payload = json.loads(Path(path).read_text())
        return cls(
            weights=payload["weights"],
            bias=payload.get("bias", 0.0),
            feature_names=payload.get("feature_names", list(FEATURE_NAMES)),
            feature_mean=payload.get("feature_mean", []),
            feature_std=payload.get("feature_std", []),
            trained_on_races=payload.get("trained_on_races", 0),
            trained_on_pairs=payload.get("trained_on_pairs", 0),
            notes=payload.get("notes", ""),
        )

    def explain(self) -> List[Tuple[str, float]]:
        """Weights, largest influence first. The point of a small model."""
        return sorted(zip(self.feature_names, self.weights),
                      key=lambda pair: -abs(pair[1]))


# ------------------------------------------------------------------ fitting


def _pairs_for_race(n: int, rng: np.random.Generator) -> np.ndarray:
    """Index pairs (i, j) with i finishing ahead of j."""
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) <= MAX_PAIRS_PER_RACE:
        return np.array(all_pairs, dtype=int)
    chosen = rng.choice(len(all_pairs), MAX_PAIRS_PER_RACE, replace=False)
    return np.array([all_pairs[k] for k in chosen], dtype=int)


def fit(
    races: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    l2: float = DEFAULT_L2,
    seed: int = 0,
    notes: str = "",
) -> RankerModel:
    """Fit on a list of races, each ``(features, finish_order)``.

    ``features`` is one row per athlete; ``finish_order`` gives their finishing
    positions. Athletes are assumed already sorted by finishing position, which
    the loader guarantees.

    Standardisation is computed once over the training races and stored in the
    model, so inference applies exactly the same transform. Recomputing it at
    prediction time from a single start list would rescale the features against
    a different population and silently change every score.
    """
    rng = np.random.default_rng(seed)

    stacked = np.vstack([f for f, _ in races if len(f)])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-8] = 1.0

    n_features = stacked.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    # Build the pair difference matrix once. Every pair is (winner - loser), so
    # the label is always 1 and the model must push the score difference
    # positive. Sign symmetry means an explicit 0-labelled set adds nothing.
    diffs = []
    for feats, _order in races:
        if len(feats) < 2:
            continue
        z = (feats - mean) / std
        pairs = _pairs_for_race(len(feats), rng)
        diffs.append(z[pairs[:, 0]] - z[pairs[:, 1]])
    if not diffs:
        raise ValueError("No races with at least two rateable athletes.")
    X = np.vstack(diffs)
    n_pairs = len(X)

    # The exponential below is clipped, but the matmul itself can still emit
    # overflow warnings on a wide feature matrix while producing a perfectly
    # usable gradient. Silencing them here keeps a real warning visible if one
    # ever appears elsewhere.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(epochs):
            margin = X @ weights
            # P(winner ahead of loser); we want this near 1.
            p = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
            grad = X.T @ (p - 1.0) / n_pairs + l2 * weights
            weights -= lr * grad

    return RankerModel(
        weights=[float(w) for w in weights],
        bias=float(bias),
        feature_names=list(FEATURE_NAMES),
        feature_mean=[float(v) for v in mean],
        feature_std=[float(v) for v in std],
        trained_on_races=len(races),
        trained_on_pairs=int(n_pairs),
        notes=notes,
    )


def pair_accuracy(model: RankerModel, races: Sequence[Tuple[np.ndarray, np.ndarray]]) -> dict:
    """Share of pairs ordered correctly, and mean log loss.

    Ties in score count as half right, which is the honest treatment: a model
    that cannot separate two athletes has not predicted their order, and
    scoring it as either fully right or fully wrong is noise.
    """
    correct = 0.0
    total = 0
    loss = 0.0
    for feats, _order in races:
        if len(feats) < 2:
            continue
        scores = model.score_many(feats)
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                total += 1
                margin = scores[i] - scores[j]
                if margin > 0:
                    correct += 1
                elif margin == 0:
                    correct += 0.5
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin))))
                loss -= math.log(max(p, 1e-12))
    if not total:
        return {"pairs": 0}
    return {
        "pairs": total,
        "pair_accuracy": correct / total,
        "log_loss": loss / total,
    }
