"""End-to-end: parse a page, store it, fit ratings, predict a start list."""
from pathlib import Path

import pytest

from xcpredict import db, evaluate, predict as predict_mod
from xcpredict.rating import elo
from xcpredict.scrape import fis

FIXTURE = Path(__file__).parent / "fixtures" / "race_46733_results.html"


@pytest.fixture()
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.sqlite")
    entries = fis.parse_race_page(FIXTURE.read_text(), race_id="46733", season=2025)
    db.save_entries(connection, entries)
    yield connection
    connection.close()


def test_round_trip_through_sqlite(conn):
    race = db.get_race(conn, "46733")
    assert race["place"] == "Ruka (FIN)"
    assert race["kind"] == "distance"
    assert len(db.race_results(conn, "46733")) == 12
    assert len(db.start_list(conn, "46733")) == 12


def test_rate_then_predict_the_stored_start_list(conn):
    model = elo.fit(conn)
    elo.save(conn, model)

    reloaded = elo.load(conn)
    assert len(reloaded.ratings) == 12

    starters = predict_mod.starters_from_db(conn, "46733", reloaded, "distance")
    prediction = predict_mod.simulate(starters, race_id="46733", n_sims=3000, seed=11)

    assert prediction.n_starters == 12
    # the athlete who won the only race in the database should be the favourite
    assert prediction.athletes[0].name == "KARLSSON Frida"
    assert prediction.unrated == []


def test_prediction_never_includes_a_non_starter(conn):
    """Drop half the field from the start list; they must vanish from the output."""
    model = elo.fit(conn)
    elo.save(conn, model)
    full = db.start_list(conn, "46733")
    keep = [row["fis_code"] for row in full[:5]]
    conn.execute("DELETE FROM start_list WHERE race_id = ? AND fis_code NOT IN "
                 f"({','.join('?' * len(keep))})", ["46733", *keep])
    conn.commit()

    starters = predict_mod.starters_from_db(conn, "46733", elo.load(conn), "distance")
    prediction = predict_mod.simulate(starters, n_sims=1000, seed=1)
    assert prediction.n_starters == 5
    assert {a.fis_code for a in prediction.athletes} == set(keep)


def test_backtest_runs_on_a_single_race(conn):
    report = evaluate.run(conn, min_rated=1)
    # nobody has a prior rating before the first race, so it is skipped, not scored
    assert report.n_skipped == 1
    assert report.n_races == 0


def test_cli_predict_smoke(conn, tmp_path, capsys):
    from xcpredict.cli import main

    elo.save(conn, elo.fit(conn))
    conn.commit()
    assert main(["--db", str(tmp_path / "test.sqlite"), "predict", "46733",
                 "--sims", "500", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "KARLSSON Frida" in out
    assert "Ruka (FIN)" in out
