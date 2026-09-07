"""Tests for the similarity kernel and the features built from it.

Two things here are worth more than the rest. The kernel has to actually rank
similarity the way the sport does — a 10 km skate is more like a 15 km skate
than like a sprint — and no feature may ever see a result from after the race
it is predicting. The second is the one that produces impressive numbers when
it breaks, which is why it gets the most tests.
"""
from __future__ import annotations

from datetime import date

import pytest

from xcpredict import features
from xcpredict.features import (
    Performance,
    build_features,
    kind_similarity,
    length_similarity,
    performances_before,
    recency_weight,
    technique_similarity,
)


def race(kind="distance", technique="F", length_km=10.0):
    """A dict that quacks like the sqlite Row the kernel is given."""
    class Row(dict):
        def keys(self):
            return super().keys()
    return Row(kind=kind, technique=technique, length_km=length_km)


def perf(when, kind="distance", technique="F", length=10.0, pct=0.2, points=20.0):
    return Performance(race_date=when, kind=kind, technique=technique,
                       length_km=length, percentile=pct, field_size=60,
                       fis_points=points, finished=True)


# ---------------------------------------------------------------- the kernel


def test_identical_technique_is_full_weight():
    assert technique_similarity("F", "F") == 1.0
    assert technique_similarity("C", "C") == 1.0


def test_different_technique_is_reduced_but_never_zero():
    w = technique_similarity("C", "F")
    assert 0 < w < 1, "a classic result is weak evidence about skate, not none"


def test_skiathlon_sits_between_the_two():
    both = technique_similarity("CF", "F")
    assert technique_similarity("C", "F") < both < 1.0


def test_unknown_technique_is_not_penalised():
    """Missing data is not a mismatch; penalising it would silently
    down-weight whole seasons that happened to be scraped without it."""
    assert technique_similarity("?", "F") == 1.0


def test_length_similarity_peaks_at_the_same_distance():
    assert length_similarity(10, 10) == pytest.approx(1.0)


def test_a_15k_is_more_like_a_10k_than_a_sprint_is():
    near = length_similarity(10, 15)
    far = length_similarity(10, 1.5)
    assert near > far
    assert far < 0.1, "a sprint should barely inform a 10 km"


def test_length_similarity_is_symmetric_in_ratio():
    """5 vs 10 is the same jump as 10 vs 20, which absolute difference misses."""
    assert length_similarity(10, 5) == pytest.approx(length_similarity(10, 20), abs=1e-9)


def test_a_50k_against_a_45k_stays_close():
    # 0.978 in practice. The point is that a 10%% difference at marathon
    # distance barely matters, unlike a 10%% difference at sprint distance.
    assert length_similarity(50, 45) > 0.97
    assert length_similarity(50, 45) > length_similarity(1.5, 1.35) - 1e-9


def test_sprint_and_distance_are_reduced_not_excluded():
    assert 0 < kind_similarity("sprint", "distance") < 1


def test_a_two_km_prologue_leans_toward_sprints():
    """The behaviour Nathan asked for, and nothing hard-codes it."""
    to_sprint = length_similarity(2.0, 1.5)
    to_ten_k = length_similarity(2.0, 10.0)
    assert to_sprint > to_ten_k
    assert to_sprint > 0.8


# --------------------------------------------------------------- causality


def test_a_future_result_carries_no_weight():
    assert recency_weight(date(2026, 3, 1), date(2026, 1, 1)) == 0.0


def test_older_results_count_less():
    recent = recency_weight(date(2026, 1, 1), date(2026, 2, 1))
    old = recency_weight(date(2023, 1, 1), date(2026, 2, 1))
    assert recent > old > 0


def test_performances_before_excludes_the_race_itself_and_later():
    history = [perf(date(2025, 1, 1)), perf(date(2026, 1, 1)), perf(date(2026, 6, 1))]
    kept = performances_before(history, date(2026, 1, 1))
    assert [p.race_date for p in kept] == [date(2025, 1, 1)]


def test_features_ignore_everything_after_the_target_date():
    """The whole model's integrity in one test."""
    past_only = [perf(date(2025, 12, 1), pct=0.10)]
    with_future = past_only + [perf(date(2026, 5, 1), pct=0.99)]

    a = build_features(past_only, race(), date(2026, 1, 15))
    b = build_features(with_future, race(), date(2026, 1, 15))
    assert a == b, "a later race changed the features of an earlier one"


# ----------------------------------------------------------------- features


def test_a_debutant_has_no_features():
    """Inventing an average would fill the field with fictional mid-packers."""
    assert build_features([], race(), date(2026, 1, 1)) is None


def test_only_future_history_also_yields_nothing():
    assert build_features([perf(date(2026, 6, 1))], race(), date(2026, 1, 1)) is None


def test_better_results_produce_a_higher_form_feature():
    idx = features.FEATURE_NAMES.index("similar_form")
    strong = build_features([perf(date(2025, 12, 1), pct=0.02)], race(), date(2026, 1, 1))
    weak = build_features([perf(date(2025, 12, 1), pct=0.95)], race(), date(2026, 1, 1))
    assert strong[idx] > weak[idx]


def test_specificity_is_positive_for_a_specialist():
    """Good at this event, ordinary elsewhere -> positive specificity."""
    idx = features.FEATURE_NAMES.index("specificity")
    history = [
        perf(date(2025, 12, 1), kind="distance", technique="F", length=10, pct=0.05),
        perf(date(2025, 12, 5), kind="sprint", technique="C", length=1.4, pct=0.90),
        perf(date(2025, 12, 9), kind="sprint", technique="F", length=1.5, pct=0.88),
    ]
    vector = build_features(history, race(kind="distance", technique="F", length_km=10), date(2026, 1, 1))
    assert vector[idx] > 0


def test_specificity_is_negative_for_someone_out_of_their_event():
    idx = features.FEATURE_NAMES.index("specificity")
    history = [
        perf(date(2025, 12, 1), kind="distance", technique="F", length=10, pct=0.85),
        perf(date(2025, 12, 5), kind="sprint", technique="F", length=1.5, pct=0.05),
        perf(date(2025, 12, 9), kind="sprint", technique="C", length=1.4, pct=0.08),
    ]
    vector = build_features(history, race(kind="distance", technique="F", length_km=10), date(2026, 1, 1))
    assert vector[idx] < 0


def test_the_feature_vector_matches_the_declared_names():
    vector = build_features([perf(date(2025, 12, 1))], race(), date(2026, 1, 1))
    assert len(vector) == len(features.FEATURE_NAMES)


def test_every_feature_is_finite():
    """A single inf silently poisons the whole fitted model."""
    import math
    history = [perf(date(2025, 12, 1), pct=0.0), perf(date(2025, 12, 2), pct=1.0)]
    for value in build_features(history, race(), date(2026, 1, 1)):
        assert math.isfinite(value)
