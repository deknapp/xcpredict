"""Freezing predictions into files a browser can read.

The site this feeds has no server. That is not a compromise: every prediction
it shows is about a race that has already been run, and the ratings behind it
only move when new results arrive. There is nothing to compute per visitor, so
computing it per visitor would be waste dressed up as architecture.

So the pipeline is: fit here, write JSON, publish the JSON. The model's own
weights ship alongside, because a page that shows predictions without showing
what produced them is asking to be taken on trust.

**What each race file contains.** The predicted order, the actual order, and
the features behind each athlete's score. The features are included
deliberately -- they let the page explain *why* it expected someone to do well,
and they let anyone check the arithmetic rather than believe the ranking.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dataset import RaceSample
from .ml import RankerModel

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("site/data")


def _spearman(predicted_rank: Sequence[int], actual_rank: Sequence[int]) -> float:
    n = len(predicted_rank)
    if n < 2:
        return float("nan")
    d2 = sum((p - a) ** 2 for p, a in zip(predicted_rank, actual_rank))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def race_payload(
    sample: RaceSample,
    model: RankerModel,
    names: Dict[str, str],
    nations: Optional[Dict[str, str]] = None,
) -> dict:
    """One race, predicted and scored against what actually happened."""
    scores = model.score_many(sample.features)

    # Athletes arrive in finishing order, so index i has actual rank i+1.
    order = np.argsort(-scores)
    predicted_rank_of = {int(idx): position + 1 for position, idx in enumerate(order)}

    athletes = []
    for i, code in enumerate(sample.fis_codes):
        athletes.append({
            "code": code,
            "name": names.get(code, code),
            "nation": (nations or {}).get(code),
            "predicted": predicted_rank_of[i],
            "actual": i + 1,
            "score": round(float(scores[i]), 4),
            "features": {
                n: round(float(v), 4)
                for n, v in zip(model.feature_names, sample.features[i])
            },
        })

    athletes.sort(key=lambda a: a["predicted"])
    predicted = [a["predicted"] for a in athletes]
    actual = [a["actual"] for a in athletes]

    # Pairwise accuracy for this race alone, so the page can say how it did
    # here rather than only quoting a global average.
    correct = total = 0
    for i in range(len(athletes)):
        for j in range(i + 1, len(athletes)):
            total += 1
            if (athletes[i]["actual"] < athletes[j]["actual"]) == (
                athletes[i]["predicted"] < athletes[j]["predicted"]
            ):
                correct += 1

    return {
        "race_id": sample.race_id,
        "date": sample.race_date.isoformat() if sample.race_date else None,
        "season": sample.season,
        "place": sample.place,
        "title": sample.title,
        "gender": sample.gender,
        "kind": sample.kind,
        "technique": sample.technique,
        "length_km": sample.length_km,
        "n_athletes": len(athletes),
        "pair_accuracy": round(correct / total, 4) if total else None,
        "spearman": round(_spearman(predicted, actual), 4),
        "winner_predicted_rank": next(
            (a["predicted"] for a in athletes if a["actual"] == 1), None
        ),
        "athletes": athletes,
    }


def write_site_data(
    samples: Sequence[RaceSample],
    model: RankerModel,
    names: Dict[str, str],
    nations: Optional[Dict[str, str]] = None,
    comparison: Optional[dict] = None,
    out_dir: Path = DEFAULT_OUT,
    holdout_seasons: Sequence[int] = (),
) -> dict:
    """Write one file per race plus an index, and return a summary.

    Races are written individually so the page loads a few kilobytes when
    someone picks one, instead of several megabytes on arrival.
    """
    races_dir = out_dir / "races"
    races_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for sample in samples:
        payload = race_payload(sample, model, names, nations)
        (races_dir / f"{sample.race_id}.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        index.append({
            "race_id": payload["race_id"],
            "date": payload["date"],
            "season": payload["season"],
            "place": payload["place"],
            "title": payload["title"],
            "gender": payload["gender"],
            "kind": payload["kind"],
            "technique": payload["technique"],
            "length_km": payload["length_km"],
            "n_athletes": payload["n_athletes"],
            "pair_accuracy": payload["pair_accuracy"],
            "winner_predicted_rank": payload["winner_predicted_rank"],
            # Whether the model was trained on this race matters to anyone
            # reading a score, so it travels with the race rather than being
            # explained once in a footnote.
            "held_out": payload["season"] in set(holdout_seasons),
        })

    index.sort(key=lambda r: (r["date"] or "", r["race_id"]), reverse=True)
    (out_dir / "index.json").write_text(json.dumps({
        "n_races": len(index),
        "holdout_seasons": list(holdout_seasons),
        "races": index,
    }, separators=(",", ":")))

    (out_dir / "model.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
    if comparison is not None:
        (out_dir / "baselines.json").write_text(
            json.dumps(comparison, indent=2) + "\n")

    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.json"))
    log.info("wrote %d races to %s (%.1f MB)", len(index), out_dir, total_bytes / 1e6)
    return {
        "races": len(index),
        "out_dir": str(out_dir),
        "megabytes": round(total_bytes / 1e6, 2),
    }
