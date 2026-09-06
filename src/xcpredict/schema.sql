-- xcpredict local store. Plain SQLite, no ORM.
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS athletes (
    fis_code      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    nation        TEXT,
    birth_year    INTEGER,
    competitor_id TEXT
);

CREATE TABLE IF NOT EXISTS races (
    race_id    TEXT PRIMARY KEY,
    event_id   TEXT,
    season     INTEGER,
    race_date  TEXT,
    place      TEXT,
    series     TEXT,
    title      TEXT,
    gender     TEXT,
    kind       TEXT,
    technique  TEXT,
    start_type TEXT,
    length_km  REAL,
    is_team    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_races_date ON races (race_date);

CREATE TABLE IF NOT EXISTS results (
    race_id    TEXT NOT NULL,
    fis_code   TEXT NOT NULL,
    rank       INTEGER,
    bib        INTEGER,
    time_s     REAL,
    diff_s     REAL,
    fis_points REAL,
    status     TEXT DEFAULT 'OK',
    PRIMARY KEY (race_id, fis_code)
);
CREATE INDEX IF NOT EXISTS idx_results_athlete ON results (fis_code);

CREATE TABLE IF NOT EXISTS start_list (
    race_id    TEXT NOT NULL,
    fis_code   TEXT NOT NULL,
    bib        INTEGER,
    start_time TEXT,
    fetched_at TEXT,
    PRIMARY KEY (race_id, fis_code)
);

-- One row per (athlete, pool). Pool is "sprint" or "distance", optionally
-- suffixed by technique, e.g. "distance:C".
CREATE TABLE IF NOT EXISTS ratings (
    fis_code     TEXT NOT NULL,
    pool         TEXT NOT NULL,
    rating       REAL NOT NULL,
    n_races      INTEGER NOT NULL DEFAULT 0,
    last_race_id TEXT,
    last_date    TEXT,
    PRIMARY KEY (fis_code, pool)
);
