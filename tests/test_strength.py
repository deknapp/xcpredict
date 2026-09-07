"""Tests for field strength and the race filter.

Once races outside the World Cup enter the data, finishing position stops
meaning the same thing everywhere. Winning a regional Continental Cup and
winning the World Cup both score a percentile of 0.0, and without an
adjustment a domestic specialist who cleans up at home outranks someone
finishing fifteenth against the world.

These tests pin the two properties that stop that happening: a weak field is
worth less, and the adjustment never reorders anyone within their own race.
"""
from __future__ import annotations

import pytest

from xcpredict import strength
from xcpredict.strength import adjust_percentile, assess, count_anchors, is_usable


# ------------------------------------------------------------------ anchors


def test_an_athlete_needs_enough_starts_to_anchor():
    starts = {"veteran": 40, "regular": 5, "novice": 4, "debutant": 0}
    anchors = count_anchors(list(starts), starts)
    assert set(anchors) == {"veteran", "regular"}, "4 starts is not yet knowledge"


def test_the_start_threshold_is_configurable():
    starts = {"a": 3, "b": 8}
    assert count_anchors(["a", "b"], starts, min_starts=3) == ["a", "b"]
    assert count_anchors(["a", "b"], starts, min_starts=10) == []


def test_an_unknown_athlete_never_anchors():
    assert count_anchors(["ghost"], {}) == []


# ------------------------------------------------------------------- usable


def test_a_race_with_too_few_known_athletes_is_rejected():
    starts = {f"a{i}": 20 for i in range(4)}
    assert not is_usable(list(starts), starts, min_anchors=5)


def test_a_race_with_enough_known_athletes_is_accepted():
    starts = {f"a{i}": 20 for i in range(6)}
    assert is_usable(list(starts), starts, min_anchors=5)


def test_a_crowded_race_of_unknowns_is_still_rejected():
    """Two hundred domestic skiers and no World Cup starters teaches nothing."""
    starts = {f"a{i}": 0 for i in range(200)}
    assert not is_usable(list(starts), starts)


# ------------------------------------------------------------------ strength


def test_too_few_anchors_is_unusable_and_scores_zero():
    field = assess([0.9, 0.9], n_starters=80)
    assert not field.usable
    assert field.strength == 0.0


def test_more_anchors_make_a_stronger_field():
    few = assess([0.7] * 6, n_starters=80)
    many = assess([0.7] * 20, n_starters=80)
    assert many.strength > few.strength


def test_better_anchors_make_a_stronger_field():
    weak = assess([0.3] * 10, n_starters=80)
    strong = assess([0.9] * 10, n_starters=80)
    assert strong.strength > weak.strength


def test_both_halves_are_required():
    """Thirty mediocre anchors is not a strong field, and neither is a
    handful of superb ones. A product demands both; an average would not."""
    many_weak = assess([0.15] * 30, n_starters=90)
    few_strong = assess([0.95] * 5, n_starters=90)
    both = assess([0.9] * 25, n_starters=90)
    assert both.strength > many_weak.strength
    assert both.strength > few_strong.strength


def test_anchor_count_saturates():
    """The twentieth World Cup skier adds far less than the sixth."""
    a = assess([0.8] * 10, n_starters=90)
    b = assess([0.8] * 20, n_starters=90)
    c = assess([0.8] * 40, n_starters=90)
    assert (b.strength - a.strength) > (c.strength - b.strength)


def test_strength_never_exceeds_a_world_cup_field():
    field = assess([1.0] * 100, n_starters=200)
    assert field.strength <= 1.0


# ---------------------------------------------------------------- adjustment


def test_a_full_strength_field_is_left_alone():
    """World Cup data must behave exactly as it did before this existed."""
    for pct in (0.0, 0.25, 0.5, 1.0):
        assert adjust_percentile(pct, 1.0) == pytest.approx(pct)


def test_winning_a_weak_race_is_worth_less_than_winning_a_strong_one():
    strong = adjust_percentile(0.0, 1.0)
    weak = adjust_percentile(0.0, 0.4)
    assert weak > strong, "a win against nobody must not score like a World Cup win"


def test_winning_a_weak_race_still_beats_losing_a_strong_one():
    """It becomes a less good result, not a bad one."""
    won_weak = adjust_percentile(0.0, 0.5)
    lost_strong = adjust_percentile(0.95, 1.0)
    assert won_weak < lost_strong


def test_a_worthless_field_collapses_everyone_to_the_back():
    assert adjust_percentile(0.0, 0.0) == pytest.approx(1.0)
    assert adjust_percentile(1.0, 0.0) == pytest.approx(1.0)


def test_the_adjustment_never_reorders_a_race():
    """It rescales how much a result counts; it must not change who beat whom."""
    raw = [0.0, 0.1, 0.4, 0.7, 1.0]
    for s in (0.2, 0.5, 0.8, 1.0):
        adjusted = [adjust_percentile(p, s) for p in raw]
        assert adjusted == sorted(adjusted), f"order changed at strength {s}"


def test_adjusted_values_stay_in_range():
    for s in (0.0, 0.3, 1.0):
        for p in (0.0, 0.5, 1.0):
            assert 0.0 <= adjust_percentile(p, s) <= 1.0


def test_strength_outside_zero_to_one_is_clamped():
    assert adjust_percentile(0.0, 5.0) == pytest.approx(0.0)
    assert adjust_percentile(0.0, -1.0) == pytest.approx(1.0)


def test_a_stronger_field_spreads_results_further_apart():
    """The point of the adjustment: a strong field discriminates more."""
    spread_strong = adjust_percentile(0.8, 1.0) - adjust_percentile(0.2, 1.0)
    spread_weak = adjust_percentile(0.8, 0.3) - adjust_percentile(0.2, 0.3)
    assert spread_strong > spread_weak


def test_the_summary_round_trips():
    field = assess([0.8] * 8, n_starters=60)
    payload = field.to_dict()
    assert payload["n_anchors"] == 8
    assert payload["usable"] is True
    assert 0.0 <= payload["strength"] <= 1.0
