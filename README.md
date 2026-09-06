# xcpredict

Predicts **FIS Cross-Country World Cup** race results from athletes' previous
results — driven by the **published start list** for the specific race, so a
prediction never contains someone who is not actually starting.

That constraint is the reason this exists. A season-long ranking will happily
tell you Klæbo wins Sunday's 20 km; it does not know he skipped the trip. The
pipeline here is deliberately ordered so the field comes from FIS itself:

```
FIS calendar → events → race pages → SQLite → Elo ratings
                                                  ↓
                        start list for race N ──→ Monte-Carlo → probabilities
```

## Status

Working scaffold. The scraper, store, rating model, simulator, backtest and CLI
are implemented and tested against a real FIS page. See
[Verifying the start-list path](#verifying-the-start-list-path) for the one
piece that cannot be confirmed until the season starts.

## Install

```bash
git clone https://github.com/deknapp/xcpredict.git
cd xcpredict
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Use

```bash
# 1. build a result history (FIS season code 2025 = the 2024/25 winter)
xcpredict scrape season 2023 2024 2025 2026

# 2. fit ratings from those results
xcpredict rate

# 3. pull the start list for an upcoming race and see who is entered
xcpredict startlist 47000

# 4. simulate that race over exactly those starters
xcpredict predict 47000 --sims 20000
```

```
Women's 10km Interval Start Classic — Ruka (FIN) 2024-11-29
pool=distance  starters=12  sims=20000

  #  Athlete                    Nat     Elo     Win  Podium   Top10  ERank
--------------------------------------------------------------------------
  1  KARLSSON Frida             SWE    1530   9.8%   29.2%   88.0%    6.0
  2  JOHAUG Therese             NOR    1525   9.8%   28.2%   87.0%    6.1
  3  SLIND Astrid Oeyre         NOR    1519   9.2%   27.7%   86.6%    6.1
  4  HENNIG Katharina           GER    1514   8.7%   26.3%   85.5%    6.3
  5  WENG Heidi                 NOR    1508   8.8%   25.8%   84.7%    6.4
  6  ANDERSSON Ebba             SWE    1503   8.5%   25.4%   84.1%    6.5
```

That sample is generated from the single race in `tests/fixtures/`, which is
why the ratings are nearly flat and the win probabilities nearly uniform — one
race of history barely separates anyone. Scrape a few seasons and the spread
becomes real.

`xcpredict backtest` replays every stored race in order, predicting each one
from ratings fitted only on earlier races, and reports pairwise accuracy, log
loss, rank correlation and how often the eventual winner was in the predicted
top three.

## How it works

**Ratings are Elo over finishing order, not over the clock.** World Cup formats
are not comparable on time: a mass start is tactical, a sprint is heats, and
course profile, altitude, snow and wax swing absolute times far more than form
does. Finishing *order* is the one signal that means the same thing at every
venue. Sprint and distance are rated in separate pools — they correlate, but
not enough to merge. Idle ratings decay toward the mean on a half-life, so a
skier who has been out injured for two seasons is not still priced on old form.

**Simulation is Plackett-Luce, tied to the ratings.** Each starter's
performance is sampled as `theta + Gumbel(0, spread)` with
`theta = rating · ln10 / 400`. At `spread = 1` the resulting head-to-head
probabilities are *exactly* the Elo probabilities, so the simulator and the
rating model cannot quietly disagree; `--spread 2` says the race is more of a
lottery than the ratings imply.

**Non-finishers do not move ratings.** A DNF is usually a crash, a broken pole
or illness, which says little about speed. They are stored, and skipped by the
fit.

**Relays and team sprints are stored but excluded** from individual ratings.

## Scraping notes

These were verified against live fis-ski.com pages rather than assumed:

- There is **no separate start-list endpoint** — `start-list.html?raceid=…`
  returns 404. FIS serves the start list from the *same*
  `results.html?sectorcode=CC&raceid=…` page before a race is run, and replaces
  it with results afterwards. One parser therefore handles both, and decides
  which it is by whether the rows carry ranks and times.
- Result rows are `<a class="table-row">` blocks whose **column order shifts
  between race formats**, and which repeat the finish time in two cells (a
  desktop and a small-screen variant). Cells are identified by their content —
  7-digit FIS code, 4-digit birth year, 3-letter nation, `mm:ss.d` time,
  `+gap` — never by position. `tests/test_parse_fis.py` runs against a real
  saved page so a FIS layout change fails a test instead of silently producing
  empty scrapes.
- Requests are cached to disk and rate-limited to one per second by default.
  Please leave it that way.

### Verifying the start-list path

The scaffold was built in the off-season, when no upcoming World Cup race had a
published start list, so the *pre-race* form of the race page could not be
fetched. Parsing is written to be content-driven rather than layout-driven for
exactly this reason, but the start-list shape still needs one live check on the
first World Cup weekend. `xcpredict startlist <raceid> --force` against a race
with a published list is the check.

## Layout

```
src/xcpredict/
  models.py        Athlete, Race, Result, StartEntry
  schema.sql/db.py SQLite store, no ORM
  scrape/http.py   cached, rate-limited fetcher
  scrape/fis.py    calendar / event / race-page parsers
  rating/elo.py    multiplayer Elo, separate sprint and distance pools
  predict.py       Monte-Carlo over a start list
  evaluate.py      walk-forward backtest
  cli.py
```

## Known gaps

- **Cold start.** A skier with no prior World Cup result sits at the default
  rating, which will misprice breakout juniors. Output flags them with `*`.
  Seeding from FIS points or continental-cup results would fix it.
- No home-venue, altitude, or course-profile terms.
- Sprint heats are modelled only through the final finishing order; the
  qualification-to-heats structure carries extra signal that is ignored.
- Weather and wax conditions are not modelled at all.

## License

MIT. Not affiliated with or endorsed by FIS. Result data belongs to FIS.
