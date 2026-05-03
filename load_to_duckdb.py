#!/usr/bin/env python3
"""
Load all garlic raw JSON files into a DuckDB database.
Run this at any time — it skips dates already loaded (idempotent).
"""
import json, duckdb
from pathlib import Path
from datetime import datetime

RAW_DIR  = Path("garlic_raw_responses")
DB_PATH  = "garlic.duckdb"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

con = duckdb.connect(DB_PATH)

con.execute("""
CREATE TABLE IF NOT EXISTS garlic_prices (
    date          DATE,
    state_id      INTEGER,
    state_name    VARCHAR,
    market_id     INTEGER,
    market_name   VARCHAR,
    variety       VARCHAR,
    grade         VARCHAR,
    arrivals      DOUBLE,
    unit_arrivals VARCHAR,
    min_price     DOUBLE,
    max_price     DOUBLE,
    modal_price   DOUBLE,
    unit_price    VARCHAR,
    source        VARCHAR
)
""")

con.execute("""
CREATE INDEX IF NOT EXISTS idx_date       ON garlic_prices (date);
CREATE INDEX IF NOT EXISTS idx_state      ON garlic_prices (state_name);
CREATE INDEX IF NOT EXISTS idx_market     ON garlic_prices (market_id);
CREATE INDEX IF NOT EXISTS idx_date_state ON garlic_prices (date, state_name);
""")

# Dates already in DB — skip them for idempotency
existing = set(
    row[0].strftime("%Y-%m-%d")
    for row in con.execute("SELECT DISTINCT date FROM garlic_prices").fetchall()
)
log(f"Dates already loaded: {len(existing)}")

files = sorted(RAW_DIR.glob("*_endpoint1.json"))
log(f"Raw files found: {len(files)}")

loaded_dates = 0
loaded_rows  = 0
errors       = 0

for f in files:
    name = f.name  # e.g. 20260430_endpoint1.json
    date_str = f"{name[0:4]}-{name[4:6]}-{name[6:8]}"

    if date_str in existing:
        continue

    try:
        data = json.loads(f.read_text())
    except Exception as e:
        log(f"  ERROR reading {f.name}: {e}")
        errors += 1
        continue

    rows = []
    for state in data.get("states", []):
        sid   = state.get("stateId")
        sname = state.get("stateName", "")
        for market in state.get("markets", []):
            mid   = market.get("marketId")
            mname = market.get("marketName", "")
            for entry in market.get("data", []):
                rows.append((
                    date_str,
                    sid,
                    sname,
                    mid,
                    mname,
                    entry.get("variety", ""),
                    entry.get("grade", ""),
                    entry.get("arrivals"),
                    entry.get("unitOfArrivals", ""),
                    entry.get("minimumPrice"),
                    entry.get("maximumPrice"),
                    entry.get("modalPrice"),
                    entry.get("unitOfPrice", ""),
                    "ep1_market_report",
                ))

    if rows:
        con.executemany("""
            INSERT INTO garlic_prices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        loaded_rows  += len(rows)
        loaded_dates += 1

    if loaded_dates % 100 == 0 and loaded_dates > 0:
        log(f"  Loaded {loaded_dates} dates, {loaded_rows:,} rows so far...")

log(f"Done. Loaded {loaded_dates} new dates, {loaded_rows:,} new rows. Errors: {errors}")

# Quick stats
total = con.execute("SELECT COUNT(*) FROM garlic_prices").fetchone()[0]
d_min = con.execute("SELECT MIN(date) FROM garlic_prices").fetchone()[0]
d_max = con.execute("SELECT MAX(date) FROM garlic_prices").fetchone()[0]
states = con.execute("SELECT COUNT(DISTINCT state_name) FROM garlic_prices").fetchone()[0]
markets = con.execute("SELECT COUNT(DISTINCT market_id) FROM garlic_prices").fetchone()[0]

log(f"\n=== DATABASE SUMMARY ===")
log(f"Total rows:    {total:,}")
log(f"Date range:    {d_min} → {d_max}")
log(f"States:        {states}")
log(f"Markets:       {markets}")
log(f"DB file:       {DB_PATH}")

con.close()
