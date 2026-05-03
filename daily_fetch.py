#!/usr/bin/env python3
"""
Incremental daily Agmarknet fetch.
Finds dates missing from garlic.duckdb and fetches them from the API.
Designed to run in CI — no interactive prompts.
"""
import requests, json, duckdb, time
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH           = "garlic.duckdb"
RAW_DIR           = Path("garlic_raw_responses")
COMMODITY_ID      = 25
COMMODITY_GROUP_ID = 6
WORKERS           = 2
TIMEOUT           = 30
RETRIES           = 3
REQUEST_DELAY     = 1.5

EP = "https://api.agmarknet.gov.in/v1/prices-and-arrivals/market-report/specific"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://agmarknet.gov.in",
    "referer": "https://agmarknet.gov.in/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}


def log(msg): print(msg, flush=True)


def get_missing_dates(lookback_days=14):
    """Return dates not yet in the DB, going back lookback_days from yesterday."""
    con = duckdb.connect(DB_PATH, read_only=True)
    existing = {
        row[0].strftime("%Y-%m-%d")
        for row in con.execute("SELECT DISTINCT date FROM garlic_prices").fetchall()
    }
    con.close()
    yesterday = date.today() - timedelta(days=1)
    start     = yesterday - timedelta(days=lookback_days)
    missing   = []
    cur = start
    while cur <= yesterday:
        if cur.strftime("%Y-%m-%d") not in existing:
            missing.append(cur)
        cur += timedelta(days=1)
    return missing


def fetch_date(d: date):
    date_str = d.strftime("%Y-%m-%d")
    raw_file = RAW_DIR / f"{d.strftime('%Y%m%d')}_endpoint1.json"
    if raw_file.exists():
        try:
            data = json.loads(raw_file.read_text())
            return d, _extract_rows(data, date_str), "cached"
        except Exception:
            pass

    params = {
        "date": date_str,
        "commodityGroupId": COMMODITY_GROUP_ID,
        "commodityId": COMMODITY_ID,
        "includeExcel": "false",
    }
    for attempt in range(RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(EP, headers=HEADERS, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1)); continue
            if r.status_code != 200:
                time.sleep(5); continue
            data = r.json()
            if data.get("states"):
                raw_file.write_text(json.dumps(data))
                return d, _extract_rows(data, date_str), "ok"
            return d, [], "empty"
        except Exception:
            time.sleep(5)
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


def load_rows(rows):
    if not rows: return 0
    con = duckdb.connect(DB_PATH)
    existing = {
        r[0].strftime("%Y-%m-%d")
        for r in con.execute("SELECT DISTINCT date FROM garlic_prices").fetchall()
    }
    new = [r for r in rows if r[0] not in existing]
    if new:
        con.executemany("INSERT INTO garlic_prices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", new)
    con.close()
    return len(new)


def run(lookback_days=14):
    RAW_DIR.mkdir(exist_ok=True)
    missing = get_missing_dates(lookback_days)
    if not missing:
        log(f"daily_fetch: nothing to fetch (last {lookback_days} days already in DB)")
        return 0

    log(f"daily_fetch: {len(missing)} dates to fetch")
    all_rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_date, d): d for d in missing}
        for future in as_completed(futures):
            d, rows, status = future.result()
            log(f"  {d}  {status}  ({len(rows)} rows)")
            all_rows.extend(rows)

    n = load_rows(all_rows)
    log(f"daily_fetch: inserted {n} new rows into DB")
    return n


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    run(lookback_days=days)
