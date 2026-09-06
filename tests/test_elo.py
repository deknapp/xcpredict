from datetime import date

from xcpredict.rating.elo import EloConfig, EloModel, expected_score


def test_expected_score_is_symmetric():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) + expected_score(1500, 1900) == 1.0


def test_400_points_is_ten_to_one():
    assert round(expected_score(1900, 1500), 4) == 0.9091


def test_winner_gains_and_loser_loses():
    model = EloModel()
    model.update_race([("a", 1), ("b", 2), ("c", 3)], pool="distance",
                      when=date(2025, 1, 1))
    a = model.get("a", "distance").rating
    c = model.get("c", "distance").rating
    assert a > 1500 > c
    assert round(a + model.get("b", "distance").rating + c, 6) == 4500.0


def test_consistent_winner_separates_from_the_field():
    model = EloModel()
    for day in range(1, 21):
        model.update_race([("fast", 1), ("mid", 2), ("slow", 3)], pool="distance",
                          when=date(2025, 1, day))
    assert model.get("fast", "distance").rating > 1700
    assert expected_score(model.get("fast", "distance").rating,
                          model.get("slow", "distance").rating) > 0.9


def test_pools_are_independent():
    model = EloModel()
    model.update_race([("a", 1), ("b", 2), ("c", 3)], pool="sprint",
                      when=date(2025, 1, 1))
    assert model.get("a", "sprint").rating > 1500
    assert model.get("a", "distance").rating == 1500


def test_idle_ratings_decay_toward_the_mean():
    model = EloModel(EloConfig(half_life_days=365))
    model.update_race([("a", 1), ("b", 2), ("c", 3)], pool="distance",
                      when=date(2025, 1, 1))
    peak = model.get("a", "distance").rating
    one_year_later = model.rating_on("a", "distance", date(2026, 1, 1))
    assert 1500 < one_year_later < peak
    assert abs(one_year_later - (1500 + (peak - 1500) / 2)) < 1.0


def test_tiny_fields_are_ignored():
    model = EloModel()
    model.update_race([("a", 1), ("b", 2)], pool="distance", when=date(2025, 1, 1))
    assert model.ratings == {}
