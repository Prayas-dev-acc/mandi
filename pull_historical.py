#!/usr/bin/env python3
"""
Pull historical Agmarknet garlic data from 2010-01-01 to 2017-06-08
(the gap before the existing DB).  Uses a thread pool for parallel requests.
Skips dates already fetched (raw JSON present).  Loads new data into DuckDB.
"""
import requests, json, duckdb, time, os
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Config ────────────────────────────────────────────────────────
START_DATE  = date(2010, 1, 1)
END_DATE    = date(2017, 6, 8)      # day before existing data starts
WORKERS       = 3                   # parallel threads (more → 429 rate-limit)
TIMEOUT       = 30                  # per-request timeout (s)
RETRIES       = 3
RETRY_DELAY   = 5                   # base seconds between retries
REQUEST_DELAY = 1.2                 # polite gap between requests per worker

RAW_DIR  = Path("garlic_raw_responses")
DB_PATH  = "garlic.duckdb"
COMMODITY_ID       = 25
COMMODITY_GROUP_ID = 6

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://agmarknet.gov.in",
    "referer": "https://agmarknet.gov.in/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
EP = "https://api.agmarknet.gov.in/v1/prices-and-arrivals/market-report/specific"

# ── Shared state ──────────────────────────────────────────────────
print_lock  = Lock()
stats = {"done": 0, "with_data": 0, "empty": 0, "error": 0, "total": 0}

def log(msg):
    with print_lock:
        print(msg, flush=True)


# ── Fetch one date ────────────────────────────────────────────────

def fetch_date(d: date) -> tuple[date, list, str]:
    """Return (date, rows, status_str).  rows=[] on empty/error."""
    date_str = d.strftime("%Y-%m-%d")
    raw_file = RAW_DIR / f"{d.strftime('%Y%m%d')}_endpoint1.json"

    # already fetched?
    if raw_file.exists():
        try:
            data = json.loads(raw_file.read_text())
            rows = _extract_rows(data, date_str)
            return d, rows, "cached"
        except Exception:
            pass  # re-fetch if corrupted

    params = {
        "date": date_str,
        "commodityGroupId": COMMODITY_GROUP_ID,
        "commodityId": COMMODITY_ID,
        "includeExcel": "false",
    }

    for attempt in range(RETRIES):
        try:
            time.sleep(REQUEST_DELAY)          # polite gap on every attempt
            r = requests.get(EP, headers=HEADERS, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)      # back off hard on rate-limit
                log(f"  [429] {date_str}  waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                time.sleep(RETRY_DELAY)
                continue
            data = r.json()
            if data.get("states"):
                raw_file.write_text(json.dumps(data))
                rows = _extract_rows(data, date_str)
                return d, rows, f"ok({len(rows)})"
            else:
                return d, [], "empty"
        except requests.exceptions.Timeout:
            time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            time.sleep(RETRY_DELAY)

    return d, [], "failed"


def _extract_rows(data, date_str):
    rows = []
    for state in data.get("states", []):
        sid, sname = state.get("stateId"), state.get("stateName", "")
        for market in state.get("markets", []):
            mid, mname = market.get("marketId"), market.get("marketName", "")
            for e in market.get("data", []):
                rows.append((
                    date_str, sid, sname, mid, mname,
                    e.get("variety", ""), e.get("grade", ""),
                    e.get("arrivals"), e.get("unitOfArrivals", ""),
                    e.get("minimumPrice"), e.get("maximumPrice"), e.get("modalPrice"),
                    e.get("unitOfPrice", ""), "ep1_market_report",
                ))
    return rows


# ── Build date list ───────────────────────────────────────────────

def build_date_list():
    existing_files = {f.name[:8] for f in RAW_DIR.glob("*_endpoint1.json")}
    dates = []
    cur = START_DATE
    while cur <= END_DATE:
        if cur.strftime("%Y%m%d") not in existing_files:
            dates.append(cur)
        cur += timedelta(days=1)
    return dates


# ── Load rows into DuckDB ─────────────────────────────────────────

def load_to_db(all_rows):
    if not all_rows:
        return 0
    con = duckdb.connect(DB_PATH)
    existing_dates = {
        row[0].strftime("%Y-%m-%d")
        for row in con.execute("SELECT DISTINCT date FROM garlic_prices").fetchall()
    }
    new_rows = [r for r in all_rows if r[0] not in existing_dates]
    if new_rows:
        con.executemany("INSERT INTO garlic_prices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", new_rows)
    con.close()
    return len(new_rows)


# ── Main ──────────────────────────────────────────────────────────

def main():
    RAW_DIR.mkdir(exist_ok=True)
    os.chdir(Path(__file__).parent)

    dates_to_fetch = build_date_list()
    stats["total"] = len(dates_to_fetch)
    total_days = (END_DATE - START_DATE).days + 1

    print(f"Historical pull: {START_DATE} → {END_DATE}  ({total_days} calendar days)")
    print(f"Already cached: {total_days - stats['total']}  |  Need to fetch: {stats['total']}")
    print(f"Workers: {WORKERS}  |  Timeout: {TIMEOUT}s  |  Retries: {RETRIES}")
    print("─" * 60)

    if not dates_to_fetch:
        print("Nothing to fetch — all dates already cached.")
        _load_all_and_report()
        return

    all_rows = []
    t0 = time.time()
    last_report = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_date, d): d for d in dates_to_fetch}

        for future in as_completed(futures):
            d, rows, status = future.result()
            stats["done"] += 1
            if status == "failed":
                stats["error"] += 1
            elif rows:
                stats["with_data"] += 1
                all_rows.extend(rows)
            else:
                stats["empty"] += 1

            # Progress every 5s
            if time.time() - last_report >= 5:
                elapsed  = time.time() - t0
                rate     = stats["done"] / elapsed
                remaining = (stats["total"] - stats["done"]) / rate if rate > 0 else 0
                log(f"  {stats['done']:4}/{stats['total']}  "
                    f"data={stats['with_data']}  empty={stats['empty']}  err={stats['error']}  "
                    f"rows_buffered={len(all_rows):,}  "
                    f"{rate:.1f} dates/s  ETA {remaining/60:.1f}min")
                last_report = time.time()

                # Flush to DB every 500 rows buffered
                if len(all_rows) >= 500:
                    n = load_to_db(all_rows)
                    log(f"  → Flushed {n:,} rows to DB")
                    all_rows.clear()

    # Final flush
    elapsed = time.time() - t0
    if all_rows:
        n = load_to_db(all_rows)
        print(f"\nFinal flush: {n:,} rows inserted")

    print(f"\n{'─'*60}")
    print(f"Fetch done in {elapsed/60:.1f} min")
    print(f"  Dates fetched : {stats['done']}")
    print(f"  With data     : {stats['with_data']}")
    print(f"  Empty/no-garlic: {stats['empty']}")
    print(f"  Errors        : {stats['error']}")

    _load_all_and_report()


def _load_all_and_report():
    """Load any remaining cached raw files not yet in DB, then print DB stats."""
    print("\nScanning cached raw files for any not yet in DB...")
    con = duckdb.connect(DB_PATH)
    existing_dates = {
        row[0].strftime("%Y-%m-%d")
        for row in con.execute(
            "SELECT DISTINCT date FROM garlic_prices WHERE date < '2017-06-09'"
        ).fetchall()
    }

    batch = []
    loaded = 0
    for f in sorted(RAW_DIR.glob("*_endpoint1.json")):
        name = f.name
        date_str = f"{name[0:4]}-{name[4:6]}-{name[6:8]}"
        if date_str >= "2017-06-09" or date_str in existing_dates:
            continue
        try:
            data = json.loads(f.read_text())
            rows = _extract_rows(data, date_str)
            batch.extend(rows)
        except Exception:
            pass
        if len(batch) >= 2000:
            con.executemany("INSERT INTO garlic_prices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            loaded += len(batch)
            batch.clear()

    if batch:
        con.executemany("INSERT INTO garlic_prices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        loaded += len(batch)

    if loaded:
        print(f"Loaded {loaded:,} additional rows from cache")

    # Refresh view if it exists
    try:
        con.execute("DROP VIEW IF EXISTS clean_garlic_prices")
        con.execute("""
            CREATE VIEW clean_garlic_prices AS
            SELECT date, market_id, market_name, state_name,
                   modal_price, arrivals
            FROM garlic_prices
            WHERE modal_price > 0 AND arrivals > 0
        """)
    except Exception:
        pass

    total  = con.execute("SELECT COUNT(*) FROM garlic_prices").fetchone()[0]
    d_min  = con.execute("SELECT MIN(date) FROM garlic_prices").fetchone()[0]
    d_max  = con.execute("SELECT MAX(date) FROM garlic_prices").fetchone()[0]
    states = con.execute("SELECT COUNT(DISTINCT state_name) FROM garlic_prices").fetchone()[0]
    markets= con.execute("SELECT COUNT(DISTINCT market_id) FROM garlic_prices").fetchone()[0]
    con.close()

    print(f"\n{'═'*60}")
    print(f"DATABASE SUMMARY")
    print(f"{'═'*60}")
    print(f"  Total rows : {total:,}")
    print(f"  Date range : {d_min} → {d_max}")
    print(f"  States     : {states}")
    print(f"  Markets    : {markets}")


if __name__ == "__main__":
    main()
