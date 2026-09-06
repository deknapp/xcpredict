import pytest

from xcpredict.predict import Starter, format_table, simulate


def _starters(*ratings):
    return [Starter(fis_code=str(i), name=f"Skier {i}", nation="USA",
                    rating=r, n_races=20)
            for i, r in enumerate(ratings, start=1)]


def test_probabilities_are_normalised():
    prediction = simulate(_starters(1800, 1600, 1500, 1400), n_sims=4000, seed=1)
    assert sum(a.p_win for a in prediction.athletes) == pytest.approx(1.0, abs=1e-9)
    assert sum(a.p_podium for a in prediction.athletes) == pytest.approx(3.0, abs=1e-9)


def test_stronger_skier_wins_more_often():
    prediction = simulate(_starters(1900, 1500), n_sims=20000, seed=7)
    best = prediction.athletes[0]
    assert best.rating == 1900
    # Gumbel sampling must reproduce Elo's own head-to-head odds (~0.909)
    assert best.p_win == pytest.approx(0.909, abs=0.02)


def test_output_is_ordered_by_expected_rank():
    prediction = simulate(_starters(1700, 1500, 1900), n_sims=4000, seed=3)
    ranks = [a.mean_rank for a in prediction.athletes]
    assert ranks == sorted(ranks)


def test_more_spread_flattens_the_favourite():
    tight = simulate(_starters(1900, 1500), n_sims=20000, seed=5, spread=1.0)
    loose = simulate(_starters(1900, 1500), n_sims=20000, seed=5, spread=3.0)
    assert loose.athletes[0].p_win < tight.athletes[0].p_win


def test_only_the_supplied_starters_appear():
    """The whole point: nobody who is not on the start list can be predicted."""
    prediction = simulate(_starters(1900, 1500, 1500), n_sims=1000, seed=2)
    assert prediction.n_starters == 3
    assert {a.fis_code for a in prediction.athletes} == {"1", "2", "3"}


def test_unrated_starters_are_flagged():
    starters = _starters(1600, 1500)
    starters[1] = Starter(fis_code="new", name="Rookie", rating=1500, n_races=0)
    prediction = simulate(starters, n_sims=1000, seed=4)
    assert prediction.unrated == ["new"]
    assert "*" in format_table(prediction)


def test_empty_start_list_is_an_error():
    with pytest.raises(ValueError):
        simulate([])
