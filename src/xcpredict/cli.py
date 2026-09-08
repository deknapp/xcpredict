"""Command line interface.

    xcpredict scrape season 2025 2026   # build the result history
    xcpredict scrape race 46733         # one race page (results or start list)
    xcpredict rate                      # fit Elo ratings
    xcpredict backtest                  # walk-forward evaluation
    xcpredict startlist 47000           # refresh a start list, show who is in
    xcpredict predict 47000             # simulate that race's start list
    xcpredict train                     # fit the learned ranker, save weights
    xcpredict evaluate                  # model vs every baseline, held out
    xcpredict export                    # freeze predictions for the web page
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import baselines, db, dataset, evaluate, export as export_mod
from . import ml, predict as predict_mod
from .rating import elo
from .scrape import fis
from .scrape.http import Fetcher


DEFAULT_MODEL = Path("data/model.json")
DEFAULT_HOLDOUT = [2026]


def _fetcher(args) -> Fetcher:
    return Fetcher(cache_dir=Path(args.cache), delay_s=args.delay,
                   use_cache=not args.no_cache)


# ------------------------------------------------------------------- commands

def cmd_scrape_season(args) -> int:
    conn = db.connect(args.db)
    fetcher = _fetcher(args)
    total = 0
    for season in args.seasons:
        categories = args.categories or ["WC"]
        for entries in fis.crawl_categories(fetcher, season, categories,
                                            force=args.force):
            db.save_entries(conn, entries)
            total += 1
            print(f"{entries.race.race_id}  {entries.race.race_date}  "
                  f"{entries.race.place or '?':<18} {entries.race.title or '?':<40} "
                  f"{len(entries.results) or len(entries.start_list)} rows")
    print(f"\nstored {total} races in {args.db}")
    return 0


def cmd_scrape_race(args) -> int:
    conn = db.connect(args.db)
    fetcher = _fetcher(args)
    for race_id in args.race_ids:
        entries = fis.crawl_race(fetcher, race_id, season=args.season, force=args.force)
        if entries is None:
            print(f"{race_id}: not found", file=sys.stderr)
            continue
        db.save_entries(conn, entries)
        kind = "results" if entries.has_results else "start list"
        print(f"{race_id}: {entries.race.title or '?'} — {kind}, "
              f"{len(entries.start_list)} athletes")
    return 0


def cmd_startlist(args) -> int:
    conn = db.connect(args.db)
    if not args.offline:
        fetcher = _fetcher(args)
        entries = fis.crawl_race(fetcher, args.race_id, force=True)
        if entries is None:
            print(f"{args.race_id}: not found", file=sys.stderr)
            return 1
        db.save_entries(conn, entries)
        if entries.has_results:
            print(f"note: {args.race_id} has already been run; "
                  f"the page now shows results, not a start list.")

    rows = db.start_list(conn, args.race_id)
    if not rows:
        print(f"no start list stored for race {args.race_id}", file=sys.stderr)
        return 1
    for row in rows:
        print(f"{row['bib'] if row['bib'] is not None else '':>4}  "
              f"{row['fis_code']:>8}  {row['name'] or '?':<28} {row['nation'] or ''}")
    print(f"\n{len(rows)} starters")
    return 0


def cmd_rate(args) -> int:
    conn = db.connect(args.db)
    config = elo.EloConfig(k_base=args.k, half_life_days=args.half_life)
    model = elo.fit(conn, config)
    elo.save(conn, model)

    pools = sorted({pool for _, pool in model.ratings})
    for pool in pools:
        ranked = sorted((r for r in model.ratings.values() if r.pool == pool),
                        key=lambda r: -r.rating)
        eligible = [r for r in ranked if r.n_races >= args.min_races]
        print(f"\n== {pool} (top {args.top}, min {args.min_races} races) ==")
        names = db.athlete_names(conn)
        for i, entry in enumerate(eligible[:args.top], start=1):
            name, nation = names.get(entry.fis_code, (entry.fis_code, ""))
            print(f"{i:>3}  {name[:28]:<29} {nation or '':<4} "
                  f"{entry.rating:>6.0f}  ({entry.n_races} races)")
    print(f"\nsaved {len(model.ratings)} ratings")
    return 0


def cmd_backtest(args) -> int:
    conn = db.connect(args.db)
    config = elo.EloConfig(k_base=args.k, half_life_days=args.half_life)
    print(evaluate.run(conn, config).summary())
    return 0


def cmd_predict(args) -> int:
    conn = db.connect(args.db)
    race = db.get_race(conn, args.race_id)
    if race is None:
        print(f"race {args.race_id} is not in the database — run "
              f"`xcpredict startlist {args.race_id}` first", file=sys.stderr)
        return 1

    model = elo.load(conn, elo.EloConfig(half_life_days=args.half_life))
    if not model.ratings:
        print("no ratings stored — run `xcpredict rate` first", file=sys.stderr)
        return 1

    pool = args.pool or elo.pool_for(race)
    when = elo._parse_date(race["race_date"])
    starters = predict_mod.starters_from_db(conn, args.race_id, model, pool, when)
    if not starters:
        print(f"no start list stored for race {args.race_id} — run "
              f"`xcpredict startlist {args.race_id}`", file=sys.stderr)
        return 1

    prediction = predict_mod.simulate(starters, race_id=args.race_id, pool=pool,
                                      n_sims=args.sims, spread=args.spread,
                                      seed=args.seed)
    print(f"{race['title'] or race['race_id']} — {race['place'] or '?'} "
          f"{race['race_date'] or ''}")
    print(f"pool={pool}  starters={prediction.n_starters}  sims={prediction.n_sims}\n")
    print(predict_mod.format_table(prediction, limit=args.top))
    return 0


# --------------------------------------------------------------------- parser

#: Applied after parsing rather than as argparse defaults — see _common_options.
GLOBAL_DEFAULTS = {
    "db": str(db.DEFAULT_DB),
    "cache": "data/cache",
    "no_cache": False,
    "delay": 1.0,
    "verbose": False,
}


def _common_options() -> argparse.ArgumentParser:
    """Options accepted either before or after the subcommand.

    Every option here uses SUPPRESS, so an unused subparser copy cannot
    overwrite a value the top-level parser already stored, and the real
    defaults are filled in by _apply_defaults() after parsing.

    They must not be given real defaults via `parser.set_defaults()`:
    `parents=` shares the *same* action objects between the top-level parser
    and every subparser, and set_defaults() mutates `action.default` in place,
    which would clear SUPPRESS everywhere and make the subparser copy clobber
    the top-level value.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS,
                        help=f"SQLite path (default: {db.DEFAULT_DB})")
    common.add_argument("--cache", default=argparse.SUPPRESS,
                        help="HTTP cache directory (default: data/cache)")
    common.add_argument("--no-cache", action="store_true", default=argparse.SUPPRESS,
                        help="bypass the HTTP cache")
    common.add_argument("--delay", type=float, default=argparse.SUPPRESS,
                        help="seconds between requests to fis-ski.com (default: 1.0)")
    common.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS)
    return common


def _apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for key, value in GLOBAL_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return _apply_defaults(build_parser().parse_args(argv))


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(prog="xcpredict", description=__doc__,
                                     parents=[common],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    scrape = sub.add_parser("scrape", help="fetch pages from fis-ski.com", parents=[common])
    scrape_sub = scrape.add_subparsers(dest="what", required=True)

    season = scrape_sub.add_parser("season", help="a whole World Cup season", parents=[common])
    season.add_argument("seasons", nargs="+", type=int,
                        help="FIS season codes, e.g. 2025 for the 2024/25 winter")
    season.add_argument("--force", action="store_true", help="re-fetch cached pages")
    season.add_argument("--categories", nargs="+", default=None,
                        metavar="CODE",
                        help="FIS category codes: WC COC WSC OWG NC FIS. "
                             "Default WC only.")
    season.set_defaults(func=cmd_scrape_season)

    race = scrape_sub.add_parser("race", help="individual races by FIS raceid", parents=[common])
    race.add_argument("race_ids", nargs="+")
    race.add_argument("--season", type=int, default=None)
    race.add_argument("--force", action="store_true")
    race.set_defaults(func=cmd_scrape_race)

    startlist = sub.add_parser("startlist", help="refresh and show a race's start list", parents=[common])
    startlist.add_argument("race_id")
    startlist.add_argument("--offline", action="store_true",
                           help="read what is already stored, do not fetch")
    startlist.set_defaults(func=cmd_startlist)

    rate = sub.add_parser("rate", help="fit and store Elo ratings", parents=[common])
    rate.add_argument("--k", type=float, default=200.0)
    rate.add_argument("--half-life", type=float, default=1825.0,
                      help="days for an idle rating to decay halfway to the mean")
    rate.add_argument("--top", type=int, default=20)
    rate.add_argument("--min-races", type=int, default=5)
    rate.set_defaults(func=cmd_rate)

    backtest = sub.add_parser("backtest", help="walk-forward evaluation", parents=[common])
    backtest.add_argument("--k", type=float, default=200.0)
    backtest.add_argument("--half-life", type=float, default=1825.0)
    backtest.set_defaults(func=cmd_backtest)

    predict = sub.add_parser("predict", help="simulate a race from its start list", parents=[common])
    predict.add_argument("race_id")
    predict.add_argument("--sims", type=int, default=20000)
    predict.add_argument("--spread", type=float, default=1.0,
                         help=">1 makes the race more random")
    predict.add_argument("--pool", choices=["sprint", "distance"], default=None)
    predict.add_argument("--half-life", type=float, default=1825.0)
    predict.add_argument("--top", type=int, default=15)
    predict.add_argument("--seed", type=int, default=None)
    predict.set_defaults(func=cmd_predict)

    train = sub.add_parser("train", help="fit the learned ranker", parents=[common])
    train.add_argument("--model", default=str(DEFAULT_MODEL))
    train.add_argument("--holdout", type=int, nargs="+", default=DEFAULT_HOLDOUT,
                       metavar="SEASON",
                       help="seasons kept out of training, for honest evaluation")
    train.add_argument("--epochs", type=int, default=ml.DEFAULT_EPOCHS)
    train.add_argument("--lr", type=float, default=ml.DEFAULT_LR)
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="model against every baseline",
                        parents=[common])
    ev.add_argument("--model", default=str(DEFAULT_MODEL))
    ev.add_argument("--holdout", type=int, nargs="+", default=DEFAULT_HOLDOUT,
                    metavar="SEASON")
    ev.set_defaults(func=cmd_evaluate)

    ex = sub.add_parser("export", help="freeze predictions for the web page",
                        parents=[common])
    ex.add_argument("--model", default=str(DEFAULT_MODEL))
    ex.add_argument("--out", default="site/data")
    ex.add_argument("--holdout", type=int, nargs="+", default=DEFAULT_HOLDOUT,
                    metavar="SEASON")
    ex.set_defaults(func=cmd_export)

    return parser


# ------------------------------------------------------- the learned ranker

def _samples(args):
    conn = db.connect(args.db)
    samples = dataset.build_samples(conn)
    if not samples:
        raise SystemExit("No rateable races. Run `xcpredict scrape season ...` first.")
    return conn, samples


def cmd_train(args) -> int:
    conn, samples = _samples(args)
    train, test = dataset.split_by_season(samples, args.holdout)
    if not train:
        raise SystemExit(f"No training races outside seasons {args.holdout}.")

    model = ml.fit(dataset.as_arrays(train), epochs=args.epochs, lr=args.lr,
                   notes=f"trained on seasons excluding {sorted(args.holdout)}")
    path = model.save(Path(args.model))

    print(f"trained on {len(train)} races, {model.trained_on_pairs:,} pairs")
    print(f"held out {len(test)} races from seasons {sorted(args.holdout)}")
    print(f"saved {path}\n")
    print("learned weights (largest influence first):")
    for name, weight in model.explain():
        print(f"   {name:22s} {weight:+.3f}")
    return 0


def cmd_evaluate(args) -> int:
    conn, samples = _samples(args)
    _, test = dataset.split_by_season(samples, args.holdout)
    if not test:
        raise SystemExit(f"No races in holdout seasons {args.holdout}.")

    model = ml.RankerModel.load(Path(args.model))
    table = baselines.compare(test, model_scorer=lambda s: model.score_many(s.features))

    print(f"Held-out seasons {sorted(args.holdout)} - {len(test)} races\n")
    print(baselines.format_comparison(table))
    print("\nThe number that matters is the gap to the baselines, not the "
          "number itself.")
    return 0


def cmd_export(args) -> int:
    from . import athletes as athletes_mod

    conn, samples = _samples(args)
    model = ml.RankerModel.load(Path(args.model))
    lookup = db.athlete_names(conn)
    names = {code: value[0] for code, value in lookup.items()}
    nations = {code: value[1] for code, value in lookup.items()}

    _, test = dataset.split_by_season(samples, args.holdout)
    comparison = baselines.compare(
        test, model_scorer=lambda s: model.score_many(s.features)) if test else None

    summary = export_mod.write_site_data(
        samples, model, names, nations, comparison,
        out_dir=Path(args.out), holdout_seasons=args.holdout,
    )
    # Athletes as first-class objects, with their whole history shipped raw so
    # the page can recompute the kernel under the reader's own filters rather
    # than only showing what was decided here.
    records = athletes_mod.collect(conn)
    people = athletes_mod.write_athlete_data(records, Path(args.out))

    # The model itself, so the browser can score a field without a server.
    model.save(Path(args.out) / "model.json")

    print(f"wrote {summary['races']} races to {summary['out_dir']} "
          f"({summary['megabytes']} MB)")
    print(f"wrote {people['athletes']} athletes ({people['megabytes']} MB)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
