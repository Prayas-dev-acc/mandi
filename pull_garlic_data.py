#!/usr/bin/env python3
"""
Agmarknet Garlic Data Puller
- Endpoint 1: market-report/specific (GET, all-India per date)
- Endpoint 2: daily-price-arrival/report (POST, filterable)
- Endpoint discovery: probes known patterns
Saves raw JSON immediately per request, combined CSV at end.
"""

import requests, json, csv, os, time, sys
from datetime import datetime, date, timedelta
from pathlib import Path

BASE_URL = "https://api.agmarknet.gov.in/v1"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://agmarknet.gov.in",
    "referer": "https://agmarknet.gov.in/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
COMMODITY_ID = 25
COMMODITY_GROUP_ID = 6

RAW_DIR = Path("garlic_raw_responses")
RAW_DIR.mkdir(exist_ok=True)

all_records = []
total_records = 0


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def save_raw(filename, data):
    path = RAW_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def get_req(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log(f"  [429 rate limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                try:
                    return r.json(), 200
                except Exception:
                    return r.text, 200
            return None, r.status_code
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return None, str(e)
    return None, "max_retries"


def post_req(url, payload, retries=4):
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                log(f"  [429 rate limit] waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 200:
                try:
                    return r.json(), 200
                except Exception:
                    return r.text, 200
            return None, r.status_code
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return None, str(e)
    return None, "max_retries"


def extract_records(data):
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "records", "report", "result", "results",
                    "prices", "arrivals", "items", "rows"]:
            val = data.get(key)
            if isinstance(val, list) and val:
                return val
        for val in data.values():
            if isinstance(val, list) and val:
                return val
    return []


# ─────────────────────────────────────────────────────────────────
# TASK 1: ENDPOINT DISCOVERY
# ─────────────────────────────────────────────────────────────────

CANDIDATE_ENDPOINTS = [
    ("GET",  "/",                                          None),
    ("GET",  "/docs",                                      None),
    ("GET",  "/openapi.json",                              None),
    ("GET",  "/health",                                    None),
    ("GET",  "/prices-and-arrivals/market-report/specific",
                {"date": "2026-04-30", "commodityGroupId": 6, "commodityId": 25, "includeExcel": "false"}),
    ("GET",  "/prices-and-arrivals/summary",
                {"date": "2026-04-30", "commodityGroupId": 6, "commodityId": 25}),
    ("GET",  "/prices-and-arrivals/trends",                {"commodityId": 25}),
    ("GET",  "/prices-and-arrivals/market-report/summary", {"date": "2026-04-30", "commodityGroupId": 6}),
    ("GET",  "/prices-and-arrivals/commodity-groups",      None),
    ("GET",  "/prices-and-arrivals/commodities",           {"groupId": 6}),
    ("GET",  "/daily-price-arrival/commodities",           None),
    ("GET",  "/daily-price-arrival/commodity-groups",      None),
    ("GET",  "/daily-price-arrival/states",                None),
    ("GET",  "/daily-price-arrival/districts",             {"stateId": 19}),
    ("GET",  "/daily-price-arrival/markets",               {"districtId": 315}),
    ("GET",  "/daily-price-arrival/grades",                {"commodityId": 25}),
    ("GET",  "/daily-price-arrival/varieties",             {"commodityId": 25}),
    ("GET",  "/daily-price-arrival/data-types",            None),
    ("GET",  "/price-trends",                              {"commodityId": 25}),
    ("GET",  "/market-profiles",                           None),
    ("GET",  "/market-profiles/list",                      None),
    ("GET",  "/commodities",                               None),
    ("GET",  "/commodity-groups",                          None),
    ("GET",  "/states",                                    None),
    ("GET",  "/districts",                                 {"stateId": 19}),
    ("GET",  "/markets",                                   {"districtId": 315}),
    ("GET",  "/historical-prices",                         {"commodityId": 25}),
    ("GET",  "/price-history",                             {"commodityId": 25}),
    ("GET",  "/arrivals/summary",                          {"commodityId": 25}),
    ("GET",  "/reports/daily",                             None),
    ("GET",  "/reports/monthly",                           None),
    ("POST", "/daily-price-arrival/report",
                {"from_date": "2026-04-01", "to_date": "2026-04-30",
                 "data_type": "100006", "group": "6", "commodity": "25",
                 "state": "[19]", "district": "[315]", "market": "[]",
                 "grade": "[]", "variety": "[]", "page": "1", "limit": "10"}),
]


def discover_endpoints():
    log("=" * 60)
    log("TASK 1: DISCOVERING ALL ENDPOINTS")
    log("=" * 60)
    working = []

    for method, path, params in CANDIDATE_ENDPOINTS:
        url = BASE_URL + path
        log(f"Testing endpoint: {method} {path}...")
        if method == "GET":
            data, status = get_req(url, params=params)
        else:
            data, status = post_req(url, params or {})

        if data is not None:
            snippet = str(data)[:100].replace("\n", " ")
            log(f"  [SUCCESS] HTTP {status} | {snippet}")
            working.append({"method": method, "path": path, "status": status})
            fname = "endpoint_discovery" + path.replace("/", "_") + ".json"
            save_raw(fname, data)
        else:
            log(f"  [FAIL] status={status}")

        time.sleep(0.5)

    log(f"\nTotal endpoints discovered: {len(working)}")
    with open("discovered_endpoints.txt", "w") as f:
        f.write(f"Discovered at: {datetime.now().isoformat()}\n\n")
        for ep in working:
            f.write(f"{ep['method']} {ep['path']} (HTTP {ep['status']})\n")

    return working


# ─────────────────────────────────────────────────────────────────
# TASK 2: REFERENCE DATA
# ─────────────────────────────────────────────────────────────────

def try_get_list(paths, params=None, label=""):
    for path in paths:
        data, status = get_req(BASE_URL + path, params=params)
        if data is not None:
            log(f"  {label} from {path}: {str(data)[:150]}")
            return data, path
        time.sleep(0.3)
    return None, None


def fetch_reference_data():
    log("=" * 60)
    log("TASK 2: FETCHING ALL REFERENCE DATA")
    log("=" * 60)
    ref = {}

    # States
    log("Fetching all states...")
    data, path = try_get_list(
        ["/daily-price-arrival/states", "/states", "/prices-and-arrivals/states"],
        label="States"
    )
    if data:
        ref["states"] = data
        save_raw("ref_states.json", data)

    # Commodity groups
    log("Fetching commodity groups...")
    data, path = try_get_list(
        ["/daily-price-arrival/commodity-groups", "/prices-and-arrivals/commodity-groups", "/commodity-groups"],
        label="Commodity groups"
    )
    if data:
        ref["commodity_groups"] = data
        save_raw("ref_commodity_groups.json", data)

    # Commodities
    log("Fetching commodities (group 6)...")
    data, path = try_get_list(
        ["/daily-price-arrival/commodities", "/prices-and-arrivals/commodities", "/commodities"],
        params={"groupId": COMMODITY_GROUP_ID},
        label="Commodities"
    )
    if data:
        ref["commodities"] = data
        save_raw("ref_commodities.json", data)

    # Grades
    log("Fetching grades for garlic...")
    data, path = try_get_list(
        ["/daily-price-arrival/grades", "/grades"],
        params={"commodityId": COMMODITY_ID},
        label="Grades"
    )
    if data:
        ref["grades"] = data
        save_raw("ref_grades.json", data)

    # Varieties
    log("Fetching varieties for garlic...")
    data, path = try_get_list(
        ["/daily-price-arrival/varieties", "/varieties"],
        params={"commodityId": COMMODITY_ID},
        label="Varieties"
    )
    if data:
        ref["varieties"] = data
        save_raw("ref_varieties.json", data)

    # Data types
    log("Fetching data types...")
    data, path = try_get_list(
        ["/daily-price-arrival/data-types", "/data-types"],
        label="Data types"
    )
    if data:
        ref["data_types"] = data
        save_raw("ref_data_types.json", data)

    # Districts per state
    state_ids = _extract_state_ids(ref.get("states"))
    if not state_ids:
        state_ids = list(range(1, 38))

    log(f"Fetching districts for {len(state_ids)} states...")
    ref["districts"] = {}
    for sid in state_ids:
        data, path = try_get_list(
            ["/daily-price-arrival/districts", "/districts"],
            params={"stateId": sid},
            label=f"Districts state={sid}"
        )
        if data:
            ref["districts"][str(sid)] = data
            save_raw(f"ref_districts_state_{sid}.json", data)
        time.sleep(0.3)

    # All markets (global)
    log("Fetching all markets (global)...")
    data, path = try_get_list(
        ["/daily-price-arrival/markets", "/markets"],
        label="Markets"
    )
    if data:
        ref["markets_all"] = data
        save_raw("ref_markets_all.json", data)

    with open("reference_data.json", "w") as f:
        json.dump(ref, f, indent=2)
    log("Reference data saved to reference_data.json")

    return ref


def _extract_state_ids(states_raw):
    ids = []
    if not states_raw:
        return ids
    items = states_raw if isinstance(states_raw, list) else (
        states_raw.get("data") or states_raw.get("states") or
        states_raw.get("result") or []
    )
    for s in items:
        if isinstance(s, dict):
            sid = s.get("id") or s.get("stateId") or s.get("state_id")
            if sid:
                ids.append(int(sid))
    return ids


def _extract_district_ids(dist_raw):
    ids = []
    if not dist_raw:
        return ids
    items = dist_raw if isinstance(dist_raw, list) else (
        dist_raw.get("data") or dist_raw.get("districts") or
        dist_raw.get("result") or []
    )
    for d in items:
        if isinstance(d, dict):
            did = d.get("id") or d.get("districtId") or d.get("district_id")
            if did:
                ids.append(int(did))
    return ids


# ─────────────────────────────────────────────────────────────────
# TASK 3a: ENDPOINT 1 — all-India per date
# ─────────────────────────────────────────────────────────────────

def flatten_endpoint1(data, date_str):
    """endpoint1 returns states[].markets[].data[] — flatten to rows."""
    rows = []
    if not isinstance(data, dict):
        return rows
    states = data.get("states", [])
    for state in states:
        state_id   = state.get("stateId", "")
        state_name = state.get("stateName", "")
        for market in state.get("markets", []):
            market_id   = market.get("marketId", "")
            market_name = market.get("marketName", "")
            for entry in market.get("data", []):
                row = {
                    "date":         date_str,
                    "stateId":      state_id,
                    "stateName":    state_name,
                    "marketId":     market_id,
                    "marketName":   market_name,
                    **entry,
                    "_source":      "ep1_market_report",
                }
                rows.append(row)
    return rows


def pull_endpoint1(start_date, end_date):
    """Iterate backward from start_date to end_date (start_date >= end_date)."""
    global total_records
    log("=" * 60)
    log("TASK 3a: ENDPOINT 1 — market-report/specific (all-India, per date)")
    log(f"  Range: {start_date} backward to {end_date}")
    log("=" * 60)

    url = BASE_URL + "/prices-and-arrivals/market-report/specific"
    cur = start_date
    dates_with_data = 0
    dates_empty = 0
    consecutive_empty = 0

    while cur >= end_date:
        date_str = cur.strftime("%Y-%m-%d")
        data, status = get_req(url, params={
            "date": date_str,
            "commodityGroupId": COMMODITY_GROUP_ID,
            "commodityId": COMMODITY_ID,
            "includeExcel": "false",
        })

        if data and isinstance(data, dict) and data.get("states"):
            rows = flatten_endpoint1(data, date_str)
            n = len(rows)
            total_records += n
            dates_with_data += 1
            consecutive_empty = 0
            log(f"Fetching garlic data for {date_str}... [{n} records] (total so far: {total_records})")
            safe = date_str.replace("-", "")
            save_raw(f"{safe}_endpoint1.json", data)
            all_records.extend(rows)
        else:
            dates_empty += 1
            consecutive_empty += 1
            # Only log every empty day for recent dates; log monthly for older
            if consecutive_empty <= 5 or cur.day == 1:
                log(f"Fetching garlic data for {date_str}... [0 records, status={status}]")

            # Stop going further back if 30 consecutive empty days past 90-day window
            if cur < date.today() - timedelta(days=90) and consecutive_empty >= 30:
                log(f"  30 consecutive empty days at {date_str} — API has no earlier data. Stopping.")
                break

        cur -= timedelta(days=1)
        time.sleep(0.4)

    log(f"Endpoint 1 done: {dates_with_data} dates with data, {dates_empty} empty")


# ─────────────────────────────────────────────────────────────────
# TASK 3b: ENDPOINT 2 — daily-price-arrival/report (POST)
# ─────────────────────────────────────────────────────────────────

def pull_endpoint2_chunk(from_date, to_date, state_ids, district_ids, label):
    global total_records
    url = BASE_URL + "/daily-price-arrival/report"
    page = 1
    chunk_records = 0

    while True:
        payload = {
            "from_date": from_date,
            "to_date":   to_date,
            "data_type": "100006",
            "group":     str(COMMODITY_GROUP_ID),
            "commodity": str(COMMODITY_ID),
            "state":     json.dumps(state_ids),
            "district":  json.dumps(district_ids),
            "market":    "[]",
            "grade":     "[]",
            "variety":   "[]",
            "page":      str(page),
            "limit":     "1000",
        }
        data, status = post_req(url, payload)
        rows = extract_records(data)

        if not rows:
            if page == 1:
                log(f"  {label} {from_date}→{to_date} p{page}: [0 records, status={status}]")
            break

        enriched = [{**r, "_source": "ep2_daily_report",
                     "_from": from_date, "_to": to_date,
                     "_state_ids": state_ids} for r in rows]
        all_records.extend(enriched)
        total_records += len(rows)
        chunk_records += len(rows)

        safe = f"{from_date.replace('-','')}_{to_date.replace('-','')}_s{'_'.join(str(s) for s in state_ids[:3])}_p{page}"
        save_raw(f"{safe}_ep2.json", data)

        log(f"  {label} {from_date}→{to_date} p{page}: [{len(rows)} records]")

        if len(rows) < 1000:
            break
        page += 1
        time.sleep(0.5)

    return chunk_records


def date_chunks(start_year, end_date, months=3):
    chunks = []
    cur = date(start_year, 1, 1)
    while cur <= end_date:
        # advance by `months`
        m = cur.month - 1 + months
        next_y = cur.year + m // 12
        next_m = m % 12 + 1
        end_chunk = date(next_y, next_m, 1) - timedelta(days=1)
        if end_chunk > end_date:
            end_chunk = end_date
        chunks.append((cur.strftime("%Y-%m-%d"), end_chunk.strftime("%Y-%m-%d")))
        cur = date(next_y, next_m, 1)
    return chunks


def pull_endpoint2(ref_data):
    log("=" * 60)
    log("TASK 3b: ENDPOINT 2 — daily-price-arrival/report (POST)")
    log("=" * 60)

    state_ids = _extract_state_ids(ref_data.get("states"))
    if not state_ids:
        # All known Indian state IDs
        state_ids = list(range(1, 38))

    today = date.today()
    chunks = date_chunks(2015, today, months=3)
    log(f"States to process: {state_ids}")
    log(f"Date chunks: {len(chunks)} × 3-month periods")

    # Pass 1: all-India (no state filter) per chunk
    log("\n--- Pass 1: All-India (no state/district filter) ---")
    for from_d, to_d in chunks:
        n = pull_endpoint2_chunk(from_d, to_d, [], [], "ALL-INDIA")
        if n:
            log(f"  ALL-INDIA {from_d}→{to_d}: [{n} records total]")
        time.sleep(0.8)

    # Pass 2: per state
    log("\n--- Pass 2: Per-state ---")
    for sid in state_ids:
        dist_raw = ref_data.get("districts", {}).get(str(sid))
        dist_ids = _extract_district_ids(dist_raw)

        # Batch districts in groups of 10
        batches = [dist_ids[i:i+10] for i in range(0, len(dist_ids), 10)] if dist_ids else [[]]

        for from_d, to_d in chunks:
            for batch in batches:
                n = pull_endpoint2_chunk(from_d, to_d, [sid], batch,
                                         f"State={sid} dists={batch[:3]}")
                time.sleep(0.6)


# ─────────────────────────────────────────────────────────────────
# TASK 4: SAVE COMBINED CSV
# ─────────────────────────────────────────────────────────────────

def save_csv():
    log("=" * 60)
    log("TASK 4: SAVING COMBINED CSV")
    log("=" * 60)

    if not all_records:
        log("No records to save.")
        return

    all_keys = []
    seen = set()
    for r in all_records:
        if isinstance(r, dict):
            for k in r:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

    with open("garlic_complete_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in all_records:
            if isinstance(r, dict):
                writer.writerow(r)

    log(f"Saved {len(all_records)} rows to garlic_complete_data.csv")
    log(f"Columns ({len(all_keys)}): {', '.join(all_keys)}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).parent)
    log("=" * 60)
    log("AGMARKNET GARLIC DATA PULLER")
    log(f"Working directory: {os.getcwd()}")
    log(f"Start time: {datetime.now()}")
    log("=" * 60)

    # Task 1
    working_endpoints = discover_endpoints()

    # Task 2
    ref_data = fetch_reference_data()

    # Task 3a: Endpoint 1 — iterate backward from 2021-12-30 (2021-12-31+ already fetched)
    pull_endpoint1(start_date=date(2021, 12, 30), end_date=date(2015, 1, 1))

    # Task 3b: Endpoint 2 (POST) with rate limit tolerance
    pull_endpoint2(ref_data)

    # Task 4
    save_csv()

    # Summary
    dates = [r.get("date") or r.get("Date") or r.get("reportDate") or r.get("_from", "")
             for r in all_records if isinstance(r, dict)]
    dates = [d for d in dates if d]
    date_min = min(dates) if dates else "unknown"
    date_max = max(dates) if dates else "unknown"

    states = set()
    markets = set()
    for r in all_records:
        if isinstance(r, dict):
            s = r.get("stateName") or r.get("state") or r.get("State")
            if s: states.add(str(s))
            m = r.get("marketName") or r.get("market") or r.get("Market")
            if m: markets.add(str(m))

    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    log(f"Total endpoints discovered: {len(working_endpoints)}")
    log(f"Total records fetched:      {total_records}")
    log(f"Date range:                 {date_min} to {date_max}")
    log(f"States covered ({len(states)}): {sorted(states)}")
    log(f"Markets covered ({len(markets)}): {sorted(markets)[:30]}{'...' if len(markets)>30 else ''}")
    log(f"Raw JSON files: {RAW_DIR}/")
    log(f"Combined CSV:   garlic_complete_data.csv")
    log(f"Reference data: reference_data.json")
    log(f"Endpoints list: discovered_endpoints.txt")


if __name__ == "__main__":
    main()
