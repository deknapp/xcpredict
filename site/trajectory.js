/* Careers plotted against each other over time.
 *
 * One line per skier, races along the x-axis in the order they happened,
 * finishing place on the y-axis with first at the top. The point is comparison:
 * two or three lines on the same axes show which of them is climbing and which
 * is drifting, which no table of results makes visible.
 *
 * Two ways to read the y-axis, because they answer different questions:
 *
 *   place     — where they finished. What people actually think in, and the
 *               default for that reason.
 *   adjusted  — where they finished as a share of the field. Fifth of ninety
 *               and fifth of twelve are the same dot on a raw axis and very
 *               different rides, and over a career that spans small national
 *               fields and full World Cups the raw axis quietly flatters
 *               whoever raced the weaker ones.
 *
 * Drawn as inline SVG rather than with a charting library: it is a few hundred
 * lines of axes and polylines, and pulling in a dependency to draw them would
 * cost more than it saves and would have to be loaded before anything appeared.
 */
(function (global) {
  "use strict";

  var PALETTE = ["#1d4ed8", "#b91c1c", "#15803d", "#a16207", "#7c3aed", "#0e7490"];

  function median(xs) {
    if (!xs.length) { return 0; }
    var s = xs.slice().sort(function (a, b) { return a - b; });
    return s[Math.floor(s.length / 2)];
  }

  /* Results a skier actually finished, in date order, after filters. */
  function series(athlete, opts) {
    var out = [];
    athlete.results.forEach(function (r) {
      if (r.rank === null || r.rank === undefined) { return; }
      if (opts.worldCupOnly && !r.world_cup) { return; }
      if (opts.kind && r.kind !== opts.kind) { return; }
      if (opts.technique && r.technique !== opts.technique) { return; }
      if (opts.season && String(r.season) !== String(opts.season)) { return; }
      // A weekend selection, when there is one, wins over the other filters:
      // the user has named the races they want to look at.
      if (opts.raceIds && !opts.raceIds[r.race_id]) { return; }
      out.push(r);
    });
    out.sort(function (a, b) { return (a.date || "").localeCompare(b.date || ""); });
    return out;
  }

  function valueOf(result, mode) {
    if (mode === "adjusted") {
      // Share of the field beaten, as a "virtual place" out of 100 so the axis
      // keeps reading like a placing rather than a percentage.
      return 1 + result.percentile * 99;
    }
    return result.rank;
  }

  /* Build the chart. `people` is [{athlete, colour}], already loaded. */
  function draw(container, people, opts) {
    opts = opts || {};
    var mode = opts.mode === "adjusted" ? "adjusted" : "place";

    var lines = people.map(function (p, i) {
      return {
        name: p.athlete.name,
        colour: PALETTE[i % PALETTE.length],
        points: series(p.athlete, opts)
      };
    }).filter(function (l) { return l.points.length; });

    if (!lines.length) {
      container.innerHTML = '<p class="empty">No results match those filters.</p>';
      return;
    }

    // A shared date axis: every race any selected skier started, in order.
    var dates = {};
    lines.forEach(function (l) {
      l.points.forEach(function (r) { if (r.date) { dates[r.date] = 1; } });
    });
    var axis = Object.keys(dates).sort();
    var xOf = {};
    axis.forEach(function (d, i) { xOf[d] = i; });

    var W = 900, H = 380, PAD_L = 46, PAD_R = 16, PAD_T = 16, PAD_B = 46;
    var innerW = W - PAD_L - PAD_R, innerH = H - PAD_T - PAD_B;

    var worst = 1;
    lines.forEach(function (l) {
      l.points.forEach(function (r) { worst = Math.max(worst, valueOf(r, mode)); });
    });
    worst = Math.ceil(worst / 10) * 10;

    function px(date) {
      return PAD_L + (axis.length < 2 ? innerW / 2
        : (xOf[date] / (axis.length - 1)) * innerW);
    }
    function py(v) {
      // 1 at the top: a smaller number is a better result.
      return PAD_T + ((v - 1) / (worst - 1 || 1)) * innerH;
    }

    var svg = ['<svg viewBox="0 0 ' + W + " " + H + '" class="chart" ' +
               'preserveAspectRatio="xMidYMid meet" role="img">'];

    // Horizontal guides, labelled with places.
    var ticks = [1, 5, 10, 20, 30, 50, 100].filter(function (t) { return t <= worst; });
    if (ticks[ticks.length - 1] !== worst) { ticks.push(worst); }
    ticks.forEach(function (t) {
      var y = py(t).toFixed(1);
      svg.push('<line x1="' + PAD_L + '" x2="' + (W - PAD_R) + '" y1="' + y +
               '" y2="' + y + '" class="grid"/>');
      svg.push('<text x="' + (PAD_L - 8) + '" y="' + (parseFloat(y) + 4) +
               '" class="ylab">' + t + "</text>");
    });

    // Season boundaries, so the eye can find winters.
    var lastSeason = null;
    axis.forEach(function (d) {
      var year = d.slice(0, 4);
      if (lastSeason !== null && year !== lastSeason) {
        var x = px(d).toFixed(1);
        svg.push('<line x1="' + x + '" x2="' + x + '" y1="' + PAD_T + '" y2="' +
                 (H - PAD_B) + '" class="seasonline"/>');
        svg.push('<text x="' + x + '" y="' + (H - PAD_B + 16) +
                 '" class="xlab">' + year + "</text>");
      }
      lastSeason = year;
    });
    if (axis.length) {
      svg.push('<text x="' + px(axis[0]) + '" y="' + (H - PAD_B + 16) +
               '" class="xlab">' + axis[0].slice(0, 4) + "</text>");
    }

    lines.forEach(function (l) {
      var pts = l.points.map(function (r) {
        return px(r.date).toFixed(1) + "," + py(valueOf(r, mode)).toFixed(1);
      });
      svg.push('<polyline points="' + pts.join(" ") + '" fill="none" stroke="' +
               l.colour + '" stroke-width="1.8" stroke-linejoin="round" opacity="0.85"/>');
      l.points.forEach(function (r) {
        svg.push('<circle cx="' + px(r.date).toFixed(1) + '" cy="' +
                 py(valueOf(r, mode)).toFixed(1) + '" r="2.6" fill="' + l.colour +
                 '"><title>' + escapeXml(l.name) + " — " + escapeXml(r.date || "") +
                 "\n" + escapeXml((r.place || "") + " " + (r.title || "")) +
                 "\nplace " + r.rank + " of " + r.field_size + "</title></circle>");
      });
    });

    svg.push("</svg>");

    var legend = lines.map(function (l) {
      var places = l.points.map(function (r) { return r.rank; });
      return '<span class="key"><i style="background:' + l.colour + '"></i>' +
        escapeXml(l.name) + ' <span class="muted">' + l.points.length +
        " races · median " + median(places) + "</span></span>";
    }).join("");

    container.innerHTML = svg.join("") + '<div class="legend">' + legend + "</div>" +
      '<p class="note">' + (mode === "adjusted"
        ? "Field-adjusted: position as a share of the field, scaled to a place out of 100. "
          + "Fifth of ninety scores far better than fifth of twelve."
        : "Raw finishing place. Note that a small national field and a full World Cup "
          + "field are not the same achievement at the same number — switch to "
          + "field-adjusted to account for that.") + "</p>";
  }

  function escapeXml(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]);
    });
  }

  global.Trajectory = { draw: draw, PALETTE: PALETTE };
})(window);
