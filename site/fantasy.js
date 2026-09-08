/* Picking the best team you can afford.
 *
 * The fantasy game gives every skier a price and you a budget. That is a
 * knapsack problem, and knapsack problems have exact answers — there is a
 * provably best roster for any budget, and it is worth computing rather than
 * eyeballing, because the interesting picks are never the obvious ones. The
 * best team is usually not the most expensive skiers you can just afford; it
 * is a couple of them plus whoever is underpriced.
 *
 * Solved by dynamic programming over a discretised budget, tracking men and
 * women separately since the roster caps are separate but the money is not.
 * Exact, not a heuristic: no greedy "best points per dollar" pass, which gets
 * this class of problem wrong whenever a cap binds.
 *
 * ── Everything the game decides, rather than the maths, lives in RULES ──
 * When Hoffman's real prices and scoring can be scraped, this is the object to
 * change and nothing else should need to move.
 */
(function (global) {
  "use strict";

  /* Four plausible shapes for how a fantasy game pays out, because the real
   * rules will not be published until later in the year and the optimiser
   * maximises whatever it is told points are. A wrong scoring rule does not
   * produce a slightly wrong team, it produces a confidently wrong one -- so
   * rather than bury one guess, these are switchable and the differences
   * between them are visible.
   *
   * The one that matters most is how deep the payout goes. A scheme paying
   * only the top ten makes stars the only thing worth buying; one paying the
   * top fifty makes cheap consistent finishers valuable and fills the roster.
   */
  var SCHEMES = {
    worldcup: {
      label: "FIS World Cup points (top 30)",
      note: "The actual World Cup table: 100 for a win, 80, 60, 50, 45, then " +
            "down to 1 for 30th. The most likely shape for a game about World " +
            "Cup racing, so it is the default.",
      points: function (place) {
        var table = [100, 80, 60, 50, 45, 40, 36, 32, 29, 26, 24, 22, 20, 18, 16,
                     15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        return place <= 30 ? table[place - 1] : 0;
      }
    },
    decay: {
      label: "Smooth decay (top 30)",
      note: "100 for a win falling by 7% a place, nothing past 30th. Gentler " +
            "at the front than the World Cup table, so second place is worth " +
            "nearly as much as first.",
      points: function (place) {
        return place > 30 ? 0 : Math.round(100 * Math.pow(0.93, place - 1));
      }
    },
    deep: {
      label: "Pays deep (top 50)",
      note: "Rewards finishing at all. Makes cheap, reliable skiers worth " +
            "buying and tends to fill the roster rather than leaving slots " +
            "empty -- worth trying to see how much the depth of the payout " +
            "changes the answer.",
      points: function (place) {
        return place > 50 ? 0 : Math.max(1, Math.round(60 - place));
      }
    },
    podium: {
      label: "Top ten only",
      note: "Brutal. Only the very front scores, so the optimiser buys stars " +
            "and ignores everyone else. Included because it shows how sharply " +
            "the best team depends on this one decision.",
      points: function (place) {
        return place > 10 ? 0 : Math.round(120 * Math.pow(0.75, place - 1));
      }
    }
  };

  var RULES = {
    /* 200 against prices of 3-30 and sixteen roster slots. Chosen so the
     * problem is actually interesting: you can afford three or four of the
     * best skiers and must fill the rest cheaply, which is where the real
     * decisions are. At 100 you cannot field a full team at all and the
     * "optimum" is just five stars, which teaches nothing. Replace with the
     * real budget when it is known. */
    budget: 200,          // total to spend across both rosters
    maxMen: 8,
    maxWomen: 8,
    sharedBudget: true,   // false would give each roster half

    /* Which scoring scheme is in force. See SCHEMES above. */
    scheme: "worldcup",

    pointsForPlace: function (place) {
      return SCHEMES[RULES.scheme].points(place);
    },

    /* Estimated price, until the real ones exist. Anchored to the model's own
     * ranking of the skier, which is the best available stand-in for what a
     * game would charge: the strongest skiers cost the most. */
    priceFor: function (rankInField, fieldSize) {
      var share = 1 - (rankInField - 1) / Math.max(1, fieldSize - 1);
      return Math.max(1, Math.round(3 + 27 * Math.pow(share, 2.2)));
    }
  };

  /* Expected points for a skier, from the model's score.
   *
   * The model produces an ordering, not a distribution, so this converts rank
   * to points directly rather than pretending to know a win probability. It is
   * the weakest link in the chain and is labelled as such in the interface. */
  function expectedPoints(rankAmongSelectable) {
    return RULES.pointsForPlace(rankAmongSelectable);
  }

  /* Exact knapsack over (budget, men used, women used).
   *
   * Budget is integer dollars, so the table is budget x 9 x 9 — small enough
   * to solve outright for any realistic field. Returns the chosen entries. */
  function solve(candidates, rules) {
    rules = rules || RULES;
    var B = rules.budget, M = rules.maxMen, W = rules.maxWomen;

    // best[b][m][w] = highest points achievable, or -1 for unreachable.
    var best = [], pick = [];
    for (var b = 0; b <= B; b++) {
      best.push([]); pick.push([]);
      for (var m = 0; m <= M; m++) {
        best[b].push(new Array(W + 1).fill(-1));
        pick[b].push([]);
        for (var w = 0; w <= W; w++) { pick[b][m].push(null); }
      }
    }
    best[0][0][0] = 0;

    candidates.forEach(function (c, idx) {
      // Descending budget so each skier is used at most once.
      for (var b = B; b >= c.price; b--) {
        for (var m = M; m >= 0; m--) {
          for (var w = W; w >= 0; w--) {
            var pm = m - (c.gender === "M" ? 1 : 0);
            var pw = w - (c.gender === "W" ? 1 : 0);
            if (pm < 0 || pw < 0) { continue; }
            var prev = best[b - c.price][pm][pw];
            if (prev < 0) { continue; }
            var value = prev + c.points;
            if (value > best[b][m][w]) {
              best[b][m][w] = value;
              pick[b][m][w] = { idx: idx, from: [b - c.price, pm, pw] };
            }
          }
        }
      }
    });

    // Best reachable cell using at most the caps.
    var top = { value: -1, at: null };
    for (var bb = 0; bb <= B; bb++) {
      for (var mm = 0; mm <= M; mm++) {
        for (var ww = 0; ww <= W; ww++) {
          if (best[bb][mm][ww] > top.value) {
            top = { value: best[bb][mm][ww], at: [bb, mm, ww] };
          }
        }
      }
    }
    if (!top.at) { return { team: [], points: 0, spent: 0 }; }

    var team = [], at = top.at;
    while (at && best[at[0]][at[1]][at[2]] > 0) {
      var step = pick[at[0]][at[1]][at[2]];
      if (!step) { break; }
      team.push(candidates[step.idx]);
      at = step.from;
    }
    team.reverse();

    return {
      team: team,
      points: top.value,
      spent: team.reduce(function (a, c) { return a + c.price; }, 0)
    };
  }

  /* What the optimiser would have to give up to include someone.
   *
   * Answers the question a player actually asks — "is he worth it?" — by
   * solving twice: once freely, once with that skier forced in. */
  function costOfForcing(candidates, code, rules) {
    var free = solve(candidates, rules);
    var forced = candidates.filter(function (c) { return c.code === code; });
    if (!forced.length) { return null; }
    var rest = candidates.filter(function (c) { return c.code !== code; });
    var r = rules || RULES;
    var reduced = {
      budget: r.budget - forced[0].price,
      maxMen: r.maxMen - (forced[0].gender === "M" ? 1 : 0),
      maxWomen: r.maxWomen - (forced[0].gender === "W" ? 1 : 0),
      pointsForPlace: r.pointsForPlace
    };
    if (reduced.budget < 0 || reduced.maxMen < 0 || reduced.maxWomen < 0) {
      return null;
    }
    var withHim = solve(rest, reduced);
    return {
      free: free.points,
      forced: withHim.points + forced[0].points,
      cost: free.points - (withHim.points + forced[0].points)
    };
  }

  global.Fantasy = {
    RULES: RULES,
    SCHEMES: SCHEMES,
    solve: solve,
    expectedPoints: expectedPoints,
    costOfForcing: costOfForcing
  };
})(window);
