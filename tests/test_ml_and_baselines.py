"""Tests for the ranker and, more importantly, for the things it is measured against.

The single most valuable test in this file is
``test_fis_points_baseline_does_not_read_the_result``. The first version of
that baseline scored 0.9651 and looked like a triumph for FIS points; it was
reading the finishing position, because ``results.fis_points`` is the points
*earned in that race*. A baseline that cheats is worse than no baseline,
because it makes a working model look useless.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from xcpredict import baselines, ml
from xcpredict.dataset import RaceSample
from xcpredict.features import FEATURE_NAMES
from xcpredict.ml import RankerModel


def race_with(feature_rows, fis_points=None):
    n = len(feature_rows)
    return RaceSample(
        race_id="r1", race_date=None, season=2025, place="Ruka", title="10km F",
        gender="M", kind="distance", technique="F", length_km=10.0,
        fis_codes=[f"a{i}" for i in range(n)],
        features=np.array(feature_rows, dtype=float),
        fis_points=fis_points if fis_points is not None else [None] * n,
    )


def separable_races(n_races=25, n_athletes=8, seed=0):
    """Races where the first feature genuinely orders the field."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_races):
        # Descending skill, plus noise on the other features.
        skill = np.linspace(1.0, 0.0, n_athletes)
        rows = rng.normal(0, 0.05, size=(n_athletes, len(FEATURE_NAMES)))
        rows[:, 0] = skill
        out.append((rows, np.arange(n_athletes)))
    return out


# ------------------------------------------------------------------ fitting


def test_the_model_learns_a_feature_that_orders_the_field():
    model = ml.fit(separable_races())
    stats = ml.pair_accuracy(model, separable_races(seed=99))
    assert stats["pair_accuracy"] > 0.95


def test_an_uninformative_feature_set_lands_near_chance():
    """Guards against a fit that looks good on noise."""
    rng = np.random.default_rng(1)
    noise = [(rng.normal(0, 1, size=(8, len(FEATURE_NAMES))), np.arange(8))
             for _ in range(40)]
    model = ml.fit(noise)
    stats = ml.pair_accuracy(model, noise)
    assert 0.3 < stats["pair_accuracy"] < 0.75


def test_fitting_needs_at_least_one_usable_race():
    with pytest.raises(ValueError):
        ml.fit([(np.zeros((1, len(FEATURE_NAMES))), np.arange(1))])


def test_scoring_is_deterministic():
    model = ml.fit(separable_races())
    feats = separable_races()[0][0]
    assert np.allclose(model.score_many(feats), model.score_many(feats))


# ------------------------------------------------------------------ storage


def test_a_model_round_trips_through_json(tmp_path):
    model = ml.fit(separable_races())
    path = model.save(tmp_path / "m.json")
    again = RankerModel.load(path)

    feats = separable_races()[0][0]
    assert np.allclose(model.score_many(feats), again.score_many(feats), atol=1e-5)


def test_the_saved_file_is_small_and_readable(tmp_path):
    """It has to be reviewable in a diff and loadable in a browser."""
    path = ml.fit(separable_races()).save(tmp_path / "m.json")
    payload = json.loads(path.read_text())

    assert path.stat().st_size < 4096
    assert payload["feature_names"] == FEATURE_NAMES
    assert len(payload["weights"]) == len(FEATURE_NAMES)


def test_standardisation_travels_with_the_model(tmp_path):
    """Recomputing it from one start list would rescale against a different
    population and silently change every score."""
    model = ml.fit(separable_races())
    payload = json.loads(model.save(tmp_path / "m.json").read_text())
    assert len(payload["feature_mean"]) == len(FEATURE_NAMES)
    assert len(payload["feature_std"]) == len(FEATURE_NAMES)


def test_explain_orders_by_influence():
    model = RankerModel(weights=[0.1, -0.9, 0.4], feature_names=["a", "b", "c"])
    assert [name for name, _ in model.explain()] == ["b", "c", "a"]


# ---------------------------------------------------------------- baselines


def test_random_baseline_sits_at_chance():
    races = [race_with(np.random.default_rng(i).normal(size=(12, len(FEATURE_NAMES))))
             for i in range(60)]
    stats = baselines.evaluate_scorer(races, baselines.random_scorer(seed=3))
    assert 0.45 < stats["pair_accuracy"] < 0.55


def test_a_perfect_scorer_reaches_one():
    races = [race_with(np.zeros((6, len(FEATURE_NAMES))))]
    stats = baselines.evaluate_scorer(races, lambda s: list(range(s.n, 0, -1)))
    assert stats["pair_accuracy"] == 1.0


def test_a_reversed_scorer_reaches_zero():
    races = [race_with(np.zeros((6, len(FEATURE_NAMES))))]
    stats = baselines.evaluate_scorer(races, lambda s: list(range(s.n)))
    assert stats["pair_accuracy"] == 0.0


def test_tied_scores_count_as_half():
    """A model that cannot separate two athletes has not predicted them."""
    races = [race_with(np.zeros((4, len(FEATURE_NAMES))))]
    stats = baselines.evaluate_scorer(races, lambda s: [1.0] * s.n)
    assert stats["pair_accuracy"] == 0.5


def test_fis_points_baseline_does_not_read_the_result():
    """The regression that matters.

    ``sample.fis_points`` holds points earned *in this race*, which are a
    direct function of finishing position. The baseline must use the feature
    built from earlier races instead. If it ever reads the race column again,
    a field ordered perfectly by in-race points will score 1.0 here.
    """
    n = 10
    rows = np.zeros((n, len(FEATURE_NAMES)))
    # Prior form is deliberately uninformative: everyone equal.
    sample = race_with(rows, fis_points=[float(i) for i in range(n)])

    stats = baselines.evaluate_scorer([sample], baselines.fis_points_scorer)
    assert stats["pair_accuracy"] == 0.5, (
        "the FIS points baseline is reading the current race's points"
    )


def test_fis_points_baseline_uses_prior_points_when_they_differ():
    n = 6
    rows = np.zeros((n, len(FEATURE_NAMES)))
    idx = FEATURE_NAMES.index("fis_points")
    rows[:, idx] = np.linspace(1.0, 0.0, n)     # best prior points first
    sample = race_with(rows, fis_points=[0.0] * n)

    stats = baselines.evaluate_scorer([sample], baselines.fis_points_scorer)
    assert stats["pair_accuracy"] == 1.0


def test_the_ablation_uses_the_no_kernel_feature():
    n = 6
    rows = np.zeros((n, len(FEATURE_NAMES)))
    rows[:, FEATURE_NAMES.index("overall_form")] = np.linspace(1.0, 0.0, n)
    stats = baselines.evaluate_scorer([race_with(rows)], baselines.recent_form_scorer)
    assert stats["pair_accuracy"] == 1.0


def test_a_declined_race_is_counted_as_skipped():
    stats = baselines.evaluate_scorer(
        [race_with(np.zeros((5, len(FEATURE_NAMES))))], lambda s: None)
    assert stats["skipped"] == 1
    assert stats["pairs"] == 0


def test_comparison_table_lists_every_baseline():
    races = [race_with(np.random.default_rng(i).normal(size=(8, len(FEATURE_NAMES))))
             for i in range(10)]
    table = baselines.compare(races)
    for name in ("random", "fis_points", "recent_form (no kernel)",
                 "similar_form (kernel, unfitted)"):
        assert name in table


def test_the_formatted_table_shows_the_model_next_to_its_floor():
    races = [race_with(np.random.default_rng(i).normal(size=(8, len(FEATURE_NAMES))))
             for i in range(10)]
    text = baselines.format_comparison(baselines.compare(races))
    assert "random" in text and "fis_points" in text
