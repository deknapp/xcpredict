"""How good was the field, and therefore what a result in it is worth.

Once races other than the World Cup enter the training data, finishing
position stops meaning the same thing everywhere. The performance measure
elsewhere in this project is percentile within the field, and under that
measure winning a regional Continental Cup scores 0.0 — exactly what winning
the World Cup scores. Add those races without adjusting and a domestic
specialist who cleans up at home outranks someone who finishes fifteenth
against the best in the world.

So each race gets a strength, and a result is read against it.

**Anchors.** A race is only usable if enough athletes in it are already known
from the World Cup — call those the anchors. Anchors do two jobs: they make
the race comparable to the rest of the data at all, and their own standing is
what the strength is estimated from. An athlete counts as an anchor once they
have a few World Cup starts, because one or two starts is not yet knowledge.

**Strength.** Two things make a field strong: how many anchors are in it, and
how good those anchors are. A race with fifteen top-twenty World Cup skiers is
a much harder race to win than one with five who normally finish sixtieth, and
both are stronger than a field with no anchors at all, which is unusable.

**What the adjustment does.** A percentile is stretched toward the front in a
strong field and compressed toward the back in a weak one. Finishing tenth
against the world is worth more than winning against nobody, and after
adjustment it scores that way.

The alternative — a hand-set table of "World Cup 1.0, Continental Cup 0.7" —
was rejected because it is wrong in both directions: a strong Norwegian
national championship is a harder race than a thin World Cup sprint, and a
category code cannot tell you that. The field can.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

#: World Cup starts before an athlete's record is worth anchoring a race with.
#: Measured rather than assumed: the correlation between our estimate of an
#: athlete and their next result climbs steeply to about ten starts and then
#: flattens (0.72 at 3-5, 0.75 at 6-9, 0.78 at 10-19). Five is the compromise
#: -- strict enough that the athlete is genuinely known, loose enough that
#: plenty of domestic races still qualify.
MIN_ANCHOR_STARTS = 5

#: Anchors a race needs before it may be used at all. Below this the strength
#: estimate is too noisy to trust and the race is not really connected to the
#: rest of the data. Five anchors give ten anchored pairs.
MIN_ANCHORS = 5

#: Anchor count at which the count-based half of strength saturates. Beyond
#: this, more World Cup skiers stop making the field meaningfully harder.
ANCHOR_SATURATION = 20.0

#: Strength assigned to a full World Cup field, by construction. Everything
#: else is measured relative to this, so a strength of 1.0 means "as strong as
#: a typical World Cup race" and the World Cup data keeps behaving exactly as
#: it did before this module existed.
REFERENCE_STRENGTH = 1.0


@dataclass(frozen=True)
class FieldStrength:
    """What we know about how hard a race was to do well in."""

    n_anchors: int
    n_starters: int
    anchor_quality: float      # 0..1, mean standing of the anchors
    strength: float            # 0..1, relative to a World Cup field
    usable: bool

    def to_dict(self) -> dict:
        return {
            "n_anchors": self.n_anchors,
            "n_starters": self.n_starters,
            "anchor_quality": round(self.anchor_quality, 4),
            "strength": round(self.strength, 4),
            "usable": self.usable,
        }


def assess(
    anchor_standings: Sequence[float],
    n_starters: int,
    *,
    min_anchors: int = MIN_ANCHORS,
) -> FieldStrength:
    """Strength of one race, from the standing of the anchors who started it.

    ``anchor_standings`` is one value per anchor in 0..1, higher meaning a
    better World Cup skier. Typically their recency-weighted form.

    The two halves are multiplied rather than averaged. A field of thirty
    anchors who are all mediocre is not a strong field, and neither is one with
    three superb skiers and nobody else; requiring both to be present is the
    behaviour we want, and a product gives it.
    """
    n_anchors = len(anchor_standings)
    if n_anchors < min_anchors:
        return FieldStrength(n_anchors, n_starters, 0.0, 0.0, usable=False)

    quality = sum(anchor_standings) / n_anchors

    # Saturating in the count: the twentieth World Cup skier adds far less than
    # the sixth. log1p rather than a hard cap so it stays smooth.
    depth = math.log1p(n_anchors) / math.log1p(ANCHOR_SATURATION)
    depth = min(1.0, depth)

    return FieldStrength(
        n_anchors=n_anchors,
        n_starters=n_starters,
        anchor_quality=quality,
        strength=min(1.0, depth * quality / max(1e-9, REFERENCE_STRENGTH)),
        usable=True,
    )


def adjust_percentile(percentile: float, strength: float) -> float:
    """Rescale a within-field percentile by how strong that field was.

    Returns a percentile on a common scale, where 0 means "as good as winning
    a full World Cup field".

    Winning a weak race does not become a bad result -- it becomes a *less
    good* one. The winner of a race at strength 0.4 lands at 0.6 rather than
    0.0, which is roughly "would have been mid-pack against the world", and
    that is the honest reading of beating a field that was not there.

    The compression is toward the *back* of the scale, so ordering within a
    race is preserved exactly. This adjustment never reorders anybody; it only
    changes how much their result counts against results from other races.
    """
    strength = max(0.0, min(1.0, strength))
    return 1.0 - strength * (1.0 - percentile)


def count_anchors(
    fis_codes: Sequence[str],
    wc_starts: Dict[str, int],
    *,
    min_starts: int = MIN_ANCHOR_STARTS,
) -> List[str]:
    """Which athletes in a field are known well enough to anchor it."""
    return [code for code in fis_codes if wc_starts.get(code, 0) >= min_starts]


def is_usable(
    fis_codes: Sequence[str],
    wc_starts: Dict[str, int],
    *,
    min_starts: int = MIN_ANCHOR_STARTS,
    min_anchors: int = MIN_ANCHORS,
) -> bool:
    """Whether a non-World-Cup race has enough known athletes to be worth using."""
    return len(count_anchors(fis_codes, wc_starts, min_starts=min_starts)) >= min_anchors
