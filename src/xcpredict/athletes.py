"""Athletes as first-class objects, not rows inside a race.

The site started as a race browser: pick a race, see the prediction. That
answers one question and leaves the more interesting one alone — *what is this
skier actually good at?* — which the model already computes internally and
never showed anybody.

So this exports, per athlete, their whole result history and a profile across a
grid of race types. The profile is the similarity kernel run against each type
in turn: it is literally the same arithmetic the model uses to decide which of
your past results matter for a given race, so a strong showing under "sprint,
classic" means the model would genuinely favour that skier there.

**The history ships raw on purpose.** It would be smaller to send the finished
features, but then the page could only display what was decided here. Sending
the results themselves lets the browser recompute the kernel under whatever
settings the reader picks — World Cup only, classic only, sprints only — which
is the difference between a page that shows answers and one that lets someone
ask questions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .dataset import is_excluded, is_world_cup, load_races, load_results, _margins, _percentile
from .rating.elo import _parse_date

log = logging.getLogger(__name__)

#: The race types an athlete is profiled against. Chosen to span what the World
#: Cup actually runs rather than to be a tidy grid: there is no 50 km classic
#: sprint, and a profile row for one would be noise.
PROFILE_TYPES = [
    {"key": "sprint_c", "label": "Sprint classic", "kind": "sprint", "technique": "C", "length_km": 1.4},
    {"key": "sprint_f", "label": "Sprint free", "kind": "sprint", "technique": "F", "length_km": 1.5},
    {"key": "d10_c", "label": "10 km classic", "kind": "distance", "technique": "C", "length_km": 10.0},
    {"key": "d10_f", "label": "10 km free", "kind": "distance", "technique": "F", "length_km": 10.0},
    {"key": "d20_c", "label": "20 km classic", "kind": "distance", "technique": "C", "length_km": 20.0},
    {"key": "d20_f", "label": "20 km free", "kind": "distance", "technique": "F", "length_km": 20.0},
    {"key": "d50", "label": "50 km", "kind": "distance", "technique": "F", "length_km": 50.0},
]


@dataclass
class AthleteRecord:
    fis_code: str
    name: str
    nation: Optional[str]
    birth_year: Optional[int]
    results: List[dict]
    #: "M", "W" or None. Taken from the races they actually started rather
    #: than from any athlete field, because FIS does not publish one here and
    #: the races do. Unambiguous in practice: no athlete in this database has
    #: ever appeared in both a men's and a women's race.
    gender: Optional[str] = None

    @property
    def wc_starts(self) -> int:
        return sum(1 for r in self.results if r["world_cup"])

    def to_dict(self) -> dict:
        return {
            "fis_code": self.fis_code,
            "name": self.name,
            "nation": self.nation,
            "gender": self.gender,
            "birth_year": self.birth_year,
            "n_results": len(self.results),
            "wc_starts": self.wc_starts,
            "results": self.results,
        }


def collect(conn, *, only_with_wc_start: bool = True) -> Dict[str, AthleteRecord]:
    """Every athlete's results, in date order, across every scraped series.

    Restricted by default to athletes who have started a World Cup. The others
    are real skiers with real results, but nobody browsing this is looking for
    them, and including all 7,776 would multiply the payload eightfold for
    people the model has nothing useful to say about.
    """
    people = {
        row["fis_code"]: row
        for row in conn.execute("SELECT fis_code, name, nation, birth_year FROM athletes")
    }

    races = {r["race_id"]: r for r in load_races(conn)}
    by_athlete: Dict[str, List[dict]] = {}
    wc_starters: set[str] = set()

    for race_id, race in races.items():
        if is_excluded(race):
            continue
        results = load_results(conn, race_id)
        if not results:
            continue
        finishers = [r for r in results if r["rank"] is not None]
        field_size = len(results)
        margins = _margins(finishers)
        world_cup = is_world_cup(race)
        race_date = _parse_date(race["race_date"])

        for entry in results:
            code = entry["fis_code"]
            if world_cup and entry["rank"] is not None:
                wc_starters.add(code)
            by_athlete.setdefault(code, []).append({
                "race_id": race_id,
                "gender": race["gender"],
                "date": race_date.isoformat() if race_date else None,
                "season": race["season"],
                "place": race["place"],
                "title": race["title"],
                "series": race["series"],
                "world_cup": world_cup,
                "kind": race["kind"],
                "technique": race["technique"],
                "length_km": race["length_km"],
                "rank": entry["rank"],
                "field_size": field_size,
                "percentile": round(_percentile(entry["rank"], field_size), 4),
                "margin": (round(margins[code], 3) if code in margins else None),
            })

    records: Dict[str, AthleteRecord] = {}
    for code, results in by_athlete.items():
        if only_with_wc_start and code not in wc_starters:
            continue
        person = people.get(code)
        results.sort(key=lambda r: (r["date"] or "", r["race_id"]))
        genders = {r["gender"] for r in results if r["gender"] in ("M", "W")}
        records[code] = AthleteRecord(
            fis_code=code,
            name=person["name"] if person else code,
            nation=person["nation"] if person else None,
            birth_year=person["birth_year"] if person else None,
            results=results,
            gender=genders.pop() if len(genders) == 1 else None,
        )

    log.info("collected %d athletes with a World Cup start", len(records))
    return records


def best_events(record: AthleteRecord) -> List[dict]:
    """How this athlete rates across each race type, by the model's own kernel.

    Uses :func:`xcpredict.features.build_features` against a synthetic race of
    each type, evaluated as of the athlete's most recent result. The number
    reported is the kernel-weighted form — 1.0 would be winning everything that
    resembles this type, 0.0 losing everything.

    ``n_similar`` travels with it because a form of 0.9 over an effective one
    and a half races is not the same claim as 0.9 over twenty, and a page that
    showed only the first number would be inviting the reader to over-read it.
    """
    from .features import FEATURE_NAMES, build_features, Performance
    from .features import kind_similarity, length_similarity, recency_weight, technique_similarity

    history = [
        Performance(
            race_date=_parse_date(r["date"]),
            kind=r["kind"],
            technique=r["technique"],
            length_km=float(r["length_km"]) if r["length_km"]
            else (1.5 if r["kind"] == "sprint" else 12.0),
            percentile=r["percentile"],
            field_size=r["field_size"],
            fis_points=None,
            finished=r["rank"] is not None,
            margin=r["margin"],
        )
        for r in record.results
    ]
    if not history:
        return []

    latest = max((p.race_date for p in history if p.race_date), default=None)
    if latest is None:
        return []

    # One day after the last race, so every result counts and nothing is
    # excluded by the strict "before" rule the causal path uses.
    from datetime import timedelta
    as_of = latest + timedelta(days=1)

    form_index = FEATURE_NAMES.index("similar_form")
    rows = []
    for spec in PROFILE_TYPES:
        class Row(dict):
            def keys(self):
                return super().keys()
        race = Row(kind=spec["kind"], technique=spec["technique"],
                   length_km=spec["length_km"])

        vector = build_features(history, race, as_of)
        if vector is None:
            continue

        # Effective number of similar races: the same weights the form uses,
        # summed. This is what stops a single lucky result reading as mastery.
        effective = 0.0
        for p in history:
            effective += (
                technique_similarity(spec["technique"], p.technique)
                * kind_similarity(spec["kind"], p.kind)
                * length_similarity(spec["length_km"], p.length_km)
                * recency_weight(p.race_date, as_of)
            )

        rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "form": round(vector[form_index], 4),
            "n_similar": round(effective, 2),
        })

    rows.sort(key=lambda r: -r["form"])
    return rows


def write_athlete_data(
    records: Dict[str, AthleteRecord],
    out_dir: Path,
    *,
    min_results: int = 3,
) -> dict:
    """One file per athlete plus a searchable index."""
    people_dir = out_dir / "athletes"
    people_dir.mkdir(parents=True, exist_ok=True)

    index = []
    written = 0
    for code, record in records.items():
        if len(record.results) < min_results:
            continue
        profile = best_events(record)
        payload = record.to_dict()
        payload["best_events"] = profile
        (people_dir / f"{code}.json").write_text(json.dumps(payload, separators=(",", ":")))
        written += 1

        wins = sum(1 for r in record.results if r["rank"] == 1)
        podiums = sum(1 for r in record.results if r["rank"] and r["rank"] <= 3)
        index.append({
            "fis_code": code,
            "name": record.name,
            "nation": record.nation,
            "gender": record.gender,
            "n_results": len(record.results),
            "wc_starts": record.wc_starts,
            "wins": wins,
            "podiums": podiums,
            "best_event": profile[0]["label"] if profile else None,
            "best_form": profile[0]["form"] if profile else None,
        })

    index.sort(key=lambda a: (-a["wc_starts"], a["name"]))
    (out_dir / "athletes.json").write_text(json.dumps({
        "n_athletes": len(index),
        "athletes": index,
    }, separators=(",", ":")))

    total = sum(f.stat().st_size for f in people_dir.glob("*.json"))
    log.info("wrote %d athletes (%.1f MB)", written, total / 1e6)
    return {"athletes": written, "megabytes": round(total / 1e6, 2)}
