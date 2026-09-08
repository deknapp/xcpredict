/* The model, ported to run in the browser.
 *
 * This is a faithful port of xcpredict/features.py and xcpredict/ml.py, and it
 * exists so the page can recompute predictions under whatever filters the
 * reader chooses rather than only showing what was decided at export time.
 * Tick "classic only" and the kernel genuinely reweights every past result and
 * rescores the field; nothing here is a lookup into precomputed answers.
 *
 * The constants below are duplicated from the Python. That is a real risk --
 * two copies of a number drift -- so they are asserted against model.json at
 * load time, which carries the feature names and the standardisation the
 * Python actually fitted. If the two ever disagree the page says so loudly
 * rather than quietly scoring with the wrong arithmetic.
 */
(function (global) {
  "use strict";

  var RECENCY_HALF_LIFE_DAYS = 150.0;
  var LENGTH_SIGMA_LOG = 0.75;
  var TECHNIQUE_MISMATCH = 0.15;
  var TECHNIQUE_COMBINED = 0.75;
  var KIND_MISMATCH = 0.30;

  var FEATURE_NAMES = [
    "similar_form", "similar_best", "overall_form", "specificity",
    "experience", "recent_starts", "fis_points", "has_fis_points",
    "consistency", "similar_margin", "has_margin"
  ];

  function techniqueSimilarity(a, b) {
    if (a === b) { return 1.0; }
    if (a === "CF" || b === "CF") { return TECHNIQUE_COMBINED; }
    if (a === "?" || b === "?") { return 1.0; }
    return TECHNIQUE_MISMATCH;
  }

  function kindSimilarity(a, b) {
    if (a === b) { return 1.0; }
    if (a === "?" || b === "?") { return 1.0; }
    return KIND_MISMATCH;
  }

  function lengthSimilarity(target, other) {
    if (!(target > 0) || !(other > 0)) { return 1.0; }
    var r = Math.log(other / target);
    return Math.exp(-(r * r) / (2 * LENGTH_SIGMA_LOG * LENGTH_SIGMA_LOG));
  }

  function days(a, b) { return (b - a) / 86400000; }

  function recencyWeight(then, now) {
    if (!then || !now) { return 1.0; }
    var d = days(then, now);
    if (d < 0) { return 0.0; }   // a later race must never inform an earlier one
    return Math.pow(0.5, d / RECENCY_HALF_LIFE_DAYS);
  }

  function assumedLength(result) {
    if (result.length_km > 0) { return result.length_km; }
    return result.kind === "sprint" ? 1.5 : 12.0;
  }

  function weightedStats(values, weights) {
    var total = 0, i;
    for (i = 0; i < weights.length; i++) { total += weights[i]; }
    if (total <= 0) { return [null, null]; }
    var mean = 0;
    for (i = 0; i < values.length; i++) { mean += values[i] * weights[i]; }
    mean /= total;
    var variance = 0;
    for (i = 0; i < values.length; i++) {
      variance += weights[i] * Math.pow(values[i] - mean, 2);
    }
    return [mean, Math.sqrt(Math.max(0, variance / total))];
  }

  /* Features for one athlete against one race.
   *
   * `results` is the athlete's raw history. `filter` decides which of it the
   * reader wants counted -- this is the whole point of shipping the history
   * rather than the finished features.
   */
  function buildFeatures(results, race, asOf, filter) {
    filter = filter || {};
    var past = [], i, r;
    for (i = 0; i < results.length; i++) {
      r = results[i];
      if (!r.date) { continue; }
      var when = new Date(r.date + "T00:00:00Z");
      if (asOf && when >= asOf) { continue; }       // causality
      if (filter.worldCupOnly && !r.world_cup) { continue; }
      if (filter.kind && r.kind !== filter.kind) { continue; }
      if (filter.technique && r.technique !== filter.technique) { continue; }
      if (r.rank === null || r.rank === undefined) { continue; }
      past.push({ r: r, when: when });
    }
    if (!past.length) { return null; }

    var targetLen = race.length_km > 0 ? race.length_km
                  : (race.kind === "sprint" ? 1.5 : 12.0);

    var weights = [], pcts = [], recencyOnly = [], allPct = [];
    var marginVals = [], marginWeights = [];

    for (i = 0; i < past.length; i++) {
      var p = past[i].r, when = past[i].when;
      var sim = techniqueSimilarity(race.technique, p.technique)
              * kindSimilarity(race.kind, p.kind)
              * lengthSimilarity(targetLen, assumedLength(p));
      var rec = recencyWeight(when, asOf);
      var w = sim * rec;
      if (w > 0) { weights.push(w); pcts.push(p.percentile); }
      recencyOnly.push(rec);
      allPct.push(p.percentile);
      if (p.margin !== null && p.margin !== undefined && w > 0) {
        marginVals.push(Math.min(p.margin, 3.0));
        marginWeights.push(w);
      }
    }
    if (!weights.length) { return null; }

    var sim = weightedStats(pcts, weights);
    var similarMean = sim[0], similarSd = sim[1];
    var overall = weightedStats(allPct, recencyOnly);
    var overallMean = overall[0] === null ? similarMean : overall[0];

    // Weighted 20th percentile, so one lucky day does not define an athlete.
    var order = pcts.map(function (v, k) { return [v, weights[k]]; })
                    .sort(function (a, b) { return a[0] - b[0]; });
    var totalW = weights.reduce(function (a, b) { return a + b; }, 0);
    var cum = 0, cutoff = 0.2 * totalW, best = order[0][0];
    for (i = 0; i < order.length; i++) {
      cum += order[i][1];
      if (cum >= cutoff) { best = order[i][0]; break; }
    }

    var recentCount = 0;
    for (i = 0; i < past.length; i++) {
      if (asOf && days(past[i].when, asOf) <= 90) { recentCount++; }
    }

    var marginFeature = 0.5, hasMargin = 0.0;
    if (marginVals.length) {
      var m = weightedStats(marginVals, marginWeights)[0];
      marginFeature = 1.0 - Math.min(1.0, m / 3.0);
      hasMargin = 1.0;
    }

    // fis_points is not carried in the athlete files; the model's own mean is
    // the honest stand-in, and has_fis_points says the value is not real.
    return [
      1.0 - similarMean,
      1.0 - best,
      1.0 - overallMean,
      overallMean - similarMean,
      Math.log1p(totalW),
      Math.log1p(recentCount),
      0.5,
      0.0,
      1.0 - Math.min(1.0, similarSd === null ? 0.5 : similarSd),
      marginFeature,
      hasMargin
    ];
  }

  function score(model, features) {
    var s = 0;
    for (var i = 0; i < features.length; i++) {
      var z = (features[i] - model.feature_mean[i]) / model.feature_std[i];
      s += z * model.weights[i];
    }
    return s + (model.bias || 0);
  }

  /* Guard against the two copies of the constants drifting apart. */
  function check(model) {
    var problems = [];
    if (model.feature_names.length !== FEATURE_NAMES.length) {
      problems.push("feature count differs from the trained model");
    } else {
      for (var i = 0; i < FEATURE_NAMES.length; i++) {
        if (model.feature_names[i] !== FEATURE_NAMES[i]) {
          problems.push("feature " + i + " is '" + model.feature_names[i] +
                        "' in the model but '" + FEATURE_NAMES[i] + "' here");
        }
      }
    }
    return problems;
  }

  global.Kernel = {
    FEATURE_NAMES: FEATURE_NAMES,
    buildFeatures: buildFeatures,
    score: score,
    check: check,
    techniqueSimilarity: techniqueSimilarity,
    kindSimilarity: kindSimilarity,
    lengthSimilarity: lengthSimilarity,
    recencyWeight: recencyWeight
  };
})(window);
