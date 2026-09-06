"""Argument-parsing rules that are easy to break by accident."""
import pytest

from xcpredict.cli import parse_args


@pytest.mark.parametrize("argv", [
    ["--db", "/tmp/X.sqlite", "predict", "46733"],
    ["predict", "46733", "--db", "/tmp/X.sqlite"],
])
def test_global_options_work_on_either_side_of_the_subcommand(argv):
    assert parse_args(argv).db == "/tmp/X.sqlite"


@pytest.mark.parametrize("argv", [
    ["-v", "scrape", "season", "2025"],
    ["scrape", "season", "2025", "-v"],
    ["scrape", "-v", "season", "2025"],
])
def test_verbose_is_accepted_at_every_level(argv):
    assert parse_args(argv).verbose is True


def test_defaults_are_filled_in():
    args = parse_args(["rate"])
    assert args.db.endswith("xcpredict.sqlite")
    assert args.cache == "data/cache"
    assert args.delay == 1.0
    assert args.verbose is False
    assert args.no_cache is False


def test_subcommand_specific_options_survive():
    args = parse_args(["predict", "46733", "--sims", "99", "--spread", "2.5"])
    assert (args.sims, args.spread, args.race_id) == (99, 2.5, "46733")


def test_seasons_are_integers():
    assert parse_args(["scrape", "season", "2025", "2026"]).seasons == [2025, 2026]
