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

## The model

Two models live here. The learned one is the point.

### The learned ranker (`xcpredict train`)

Elo gives every athlete a single number, and that is its central weakness: a
sprinter and a 50 km specialist are not the same athlete, and pooling their
results discards most of what those results say.

So the model predicts an athlete in a race from what they have done in **races
like that one**. Every past result is weighted against the target race:

```
weight = technique_similarity x kind_similarity x length_similarity x recency
```

A 10 km skate leans hardest on other 10 km skates, then 15 km skates, then
10 km classics, and barely on sprints. Nothing is discarded — a sprint result
is weak evidence about a 10 km, not zero evidence — and nothing is hard-coded:
a 2 km prologue ends up near the sprints because its *length* is near theirs,
not because a rule says so.

Those weighted summaries become nine features, and a pairwise logistic ranker
is fitted on every within-race pair by gradient descent. The whole model is
nine numbers in [`data/model.json`](data/model.json) — small enough to read in
a diff and to run in a browser.

### What it actually scores

Held out on the **2026 season, never seen in training** — 76 races, 225,590
pairs:

| method | pairwise accuracy |
|---|---|
| random | 0.4986 |
| FIS points (the free alternative) | 0.6953 |
| recent form, **no similarity kernel** | 0.7750 |
| similarity kernel alone, **unfitted** | 0.8188 |
| **learned ranker** | **0.8241** |

Read that table downward, because the interesting result is not the top line.

**The similarity kernel is what works.** Weighting history by technique,
discipline and distance is worth **+4.4 points** over ignoring them
(0.7750 → 0.8188). That is the whole thesis of the model and it holds up.

**The learning on top is worth very little** — the fitted weights beat the
single best feature by a few thousandths. The features are strongly collinear,
all measuring some version of "how good is this skier at this sort of race",
so a linear combination cannot improve much on the best one alone, and
training for 8,000 epochs instead of 300 changes nothing. Reported because it
is true, and because it says where the gains actually came from.

**They came from the kernel and the data**, not the model:

| change | held-out 2026 |
|---|---|
| starting point | 0.8190 |
| tuned kernel constants | 0.8234 |
| + races beyond the World Cup | **0.8241** |

The tuning is the more interesting half. Three constants were guesses, fitted
on a validation season with the test season untouched:

| constant | guessed | tuned | what it says |
|---|---|---|---|
| technique mismatch | 0.45 | **0.15** | technique matters far more than assumed |
| recency half-life | 400 d | **150 d** | form moves faster than a season and a half |
| length sigma | 0.50 | **0.75** | a broader window over distance is better |

**Splitting the model by discipline does not help.** Fitting separate rankers
for sprint and distance scored 0.8190 against the shared model's 0.8190 —
identical. The kernel already encodes the distinction in the features, so the
weights have nothing left to separate.

### One trap worth documenting

The first version of the FIS points baseline scored **0.9651** and appeared to
demolish the model. It was cheating. `results.fis_points` holds the points
*earned in that race* — the winner scores 0.0 and it rises monotonically with
finishing position — so ordering by it is reading the answer. The baseline now
uses the points an athlete carried in from earlier races, which is what someone
consulting the FIS list before the start would have, and there is a regression
test named for it.

A baseline that cheats is worse than no baseline, because it makes a working
model look useless.

### Racing beyond the World Cup

The World Cup alone leaves **455 of 934 athletes with fewer than ten starts**,
and for those the kernel has almost nothing to weight. World Championships,
Olympics, national championships and continental series fill that in.

The obvious way to do it is wrong. Percentile within the field means winning a
regional race scores 0.0 — exactly what winning the World Cup scores — so a
domestic specialist would outrank someone finishing fifteenth against the
world.

So each race is measured rather than labelled:

* A race outside the World Cup is **usable only if five of its starters have
  five or more World Cup starts behind them**. Those anchors connect it to the
  rest of the data and are what its strength is estimated from.
* **Strength is the product** of how many anchors started and how good they
  are — a product, because thirty mediocre anchors is not a strong field, and
  neither is a handful of superb ones.
* Percentiles are rescaled by strength, so winning a weak race becomes a *less
  good* result rather than a bad one. The rescaling never reorders anyone
  inside their own race.

A hand-set table — World Cup 1.0, Continental Cup 0.7 — was rejected because it
is wrong in both directions. A strong Norwegian national championship is a
harder race than a thin World Cup sprint, and a category code cannot know that.

The five-start threshold is measured: the correlation between our estimate of
an athlete and their next result climbs steeply to about ten starts and then
flattens (0.72 at 3–5, 0.75 at 6–9, 0.78 at 10–19).

### The Elo model

Still here, still tested, and no longer the headline. It is a useful reference
point: no features, no context, one number per athlete.

## Status

Working, and backtested on real data. The scraper, store, rating model,
simulator, backtest and CLI are implemented and tested. See
[Verifying the start-list path](#verifying-the-start-list-path) for the one
piece that cannot be confirmed until the season starts.

**Backtest, 2023–2026 World Cup: 346 races, 23,320 results, 1,388 athletes.**
Every race is predicted from ratings fitted only on races that happened before
it, 297 scored:

| metric | value |
|---|---|
| pairwise accuracy | 0.800 |
| pairwise log loss | 0.4699 |
| mean rank correlation | 0.758 |
| winner in predicted top 3 | 0.576 |

`k` and the decay half-life were swept against pairwise log loss, and the
result says something about the model rather than just picking a number.
Because a race's summed surprise is normalised by the size of the field, `k` is
not the usual per-game Elo constant — it is the most a rating can move in one
race. Read that way the old default of 24 was far too conservative: a 60-skier
race is 1,770 head-to-head comparisons and was being allowed to say about as
much as one chess game. Raising it to 200 takes log loss from 0.5438 to 0.4699.

The half-life is the more interesting result, because the honest answer is
that it cannot be fitted yet. Every shorter half-life scores worse on every
metric, all the way down to switching decay off entirely — but the record is
only about three and a half seasons, so any half-life near the length of the
window is indistinguishable from no decay. Concluding "decay is harmful" would
be fitting the shortness of the dataset, not the sport. The default is five
years: long enough to cost almost nothing today, while keeping the mechanism
for the athlete who really has been out since 2023. It should be re-tuned when
there are more seasons.

One caveat on all four numbers: `k` was selected against this same walk-forward
series. Each individual race is predicted out-of-sample, but the hyperparameter
is not, so treat these as the model's ceiling rather than as a clean held-out
estimate.

## Install

```bash
git clone https://github.com/deknapp/xcpredict.git
cd xcpredict
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Use

### The learned ranker

```bash
# 1. build a result history (FIS season code 2025 = the 2024/25 winter).
#    Categories beyond the World Cup are worth pulling: they are where the
#    thinly-raced half of the field actually competes.
xcpredict scrape season 2023 2024 2025 2026 --categories WC WSC OWG NC

# 2. fit the ranker. Weights land in data/model.json -- nine numbers,
#    readable in a diff.
xcpredict train

# 3. score it against every baseline on the held-out season
xcpredict evaluate

# 4. freeze predictions for the web page
xcpredict export
```

`xcpredict evaluate` prints the model next to its floor and its competitors,
because a pairwise accuracy on its own means nothing:

```
method                              races     pairs  pair acc
--------------------------------------------------------------
random                                 76   225,590    0.4986
fis_points                             76   225,590    0.6953
recent_form (no kernel)                76   225,590    0.7750
similar_form (kernel, unfitted)        76   225,590    0.8188
learned model                          76   225,590    0.8241
```

### The Elo model

```bash
# 1. build a result history
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
venue. Ratings are pooled by gender and discipline — four pools. Sprint and
distance correlate but not enough to merge, and men and women never start
together, so every comparison the model makes is inside one gender already. Idle ratings decay toward the mean on a half-life, so a
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
