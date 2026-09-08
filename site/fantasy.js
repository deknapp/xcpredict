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

    /* Points for a finishing position. A guess at a common shape: steep at the
     * front, nothing outside the top thirty. Replace wholesale when the real
     * rules are known — the optimiser maximises whatever this returns, so a
     * wrong scoring rule produces a confidently wrong team. */
    pointsForPlace: function (place) {
      if (place > 30) { return 0; }
      return Math.round(100 * Math.pow(0.93, place - 1));
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
    solve: solve,
    expectedPoints: expectedPoints,
    costOfForcing: costOfForcing
  };
})(window);
