/* Spreadsheets, built in the browser.
 *
 * Three shapes people actually want: one race, one skier's whole career, or
 * everything from a race weekend. Built here and handed over as a download,
 * so the static site can do it with no server.
 *
 * Two details that decide whether the file is usable or annoying:
 *
 *   Excel and dates. Excel will happily reinterpret 2026-03-22 as a date in
 *   whatever the local format is, and 10-11 as the 10th of November. Every
 *   field is quoted, which stops the worst of it, and dates stay ISO because
 *   that is what any other tool wants.
 *
 *   Excel and UTF-8. Without a byte-order mark Excel reads a UTF-8 CSV as
 *   Latin-1 and turns every Scandinavian name into mojibake — Klæbo becomes
 *   KlÃ¦bo. The BOM is three bytes and fixes it, and other tools ignore it.
 */
(function (global) {
  "use strict";

  var BOM = "﻿";

  function cell(v) {
    if (v === null || v === undefined) { return '""'; }
    return '"' + String(v).replace(/"/g, '""') + '"';
  }

  function toCsv(headers, rows) {
    var out = [headers.map(cell).join(",")];
    rows.forEach(function (r) { out.push(r.map(cell).join(",")); });
    return BOM + out.join("\r\n") + "\r\n";
  }

  function download(filename, text) {
    var blob = new Blob([text], { type: "text/csv;charset=utf-8;" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Revoking immediately can cancel the download in some browsers.
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
  }

  var ATHLETE_HEADERS = [
    "date", "season", "series", "world_cup", "place", "title",
    "kind", "technique", "length_km", "rank", "field_size",
    "pct_of_field_beaten", "margin_to_winner"
  ];

  function athleteRows(athlete) {
    return athlete.results.map(function (r) {
      return [
        r.date, r.season, r.series, r.world_cup ? "yes" : "no",
        r.place, r.title, r.kind, r.technique, r.length_km,
        r.rank === null ? "" : r.rank, r.field_size,
        r.rank === null ? "" : (1 - r.percentile).toFixed(4),
        r.margin === null || r.margin === undefined ? "" : r.margin
      ];
    });
  }

  function athleteCsv(athlete) {
    return toCsv(["athlete", "nation", "gender"].concat(ATHLETE_HEADERS),
      athleteRows(athlete).map(function (row) {
        return [athlete.name, athlete.nation, athlete.gender].concat(row);
      }));
  }

  var RACE_HEADERS = [
    "predicted_rank", "actual_rank", "athlete", "nation",
    "model_score", "similar_form", "similar_races"
  ];

  function raceCsv(race) {
    var rows = race.athletes.map(function (a) {
      var f = a.features || {};
      return [
        a.predicted, a.actual, a.name, a.nation,
        a.score,
        f.similar_form === undefined ? "" : f.similar_form,
        f.experience === undefined ? "" : (Math.exp(f.experience) - 1).toFixed(2)
      ];
    });
    return toCsv(
      ["race_id", "date", "place", "title", "kind", "technique", "length_km"]
        .concat(RACE_HEADERS),
      rows.map(function (row) {
        return [race.race_id, race.date, race.place, race.title,
                race.kind, race.technique, race.length_km].concat(row);
      })
    );
  }

  /* Every race sharing a place within a few days of each other — which is what
   * a "weekend" is, since FIS does not label them and a World Cup stop can run
   * Friday to Sunday or Saturday to Monday. */
  function weekendsFrom(raceIndex) {
    var byPlace = {};
    raceIndex.forEach(function (r) {
      if (!r.date || !r.place) { return; }
      (byPlace[r.place] = byPlace[r.place] || []).push(r);
    });

    var weekends = [];
    Object.keys(byPlace).forEach(function (place) {
      var races = byPlace[place].slice().sort(function (a, b) {
        return a.date.localeCompare(b.date);
      });
      var group = [];
      races.forEach(function (r) {
        if (!group.length) { group = [r]; return; }
        var gap = (new Date(r.date) - new Date(group[group.length - 1].date)) / 86400000;
        if (gap <= 4) { group.push(r); return; }
        weekends.push({ place: place, races: group });
        group = [r];
      });
      if (group.length) { weekends.push({ place: place, races: group }); }
    });

    weekends.sort(function (a, b) {
      return b.races[0].date.localeCompare(a.races[0].date);
    });
    return weekends.map(function (w) {
      return {
        key: w.place + "|" + w.races[0].date,
        place: w.place,
        start: w.races[0].date,
        end: w.races[w.races.length - 1].date,
        race_ids: w.races.map(function (r) { return r.race_id; }),
        n_races: w.races.length
      };
    });
  }

  function weekendCsv(weekend, races) {
    var rows = [];
    races.forEach(function (race) {
      race.athletes.forEach(function (a) {
        rows.push([
          race.date, race.place, race.title, race.kind, race.technique,
          race.length_km, a.predicted, a.actual, a.name, a.nation, a.score
        ]);
      });
    });
    return toCsv(
      ["date", "place", "title", "kind", "technique", "length_km",
       "predicted_rank", "actual_rank", "athlete", "nation", "model_score"],
      rows
    );
  }

  function safeName(s) {
    return String(s || "export").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-|-$/g, "");
  }

  global.Csv = {
    toCsv: toCsv,
    download: download,
    athleteCsv: athleteCsv,
    raceCsv: raceCsv,
    weekendsFrom: weekendsFrom,
    weekendCsv: weekendCsv,
    safeName: safeName
  };
})(window);
