#!/usr/bin/env python3
"""
Merge all saved endpoint1 raw JSON files into garlic_complete_data.csv
"""
import json, csv, os
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("garlic_raw_responses")

def flatten_endpoint1(data, date_str):
    rows = []
    if not isinstance(data, dict):
        return rows
    for state in data.get("states", []):
        state_id   = state.get("stateId", "")
        state_name = state.get("stateName", "")
        for market in state.get("markets", []):
            market_id   = market.get("marketId", "")
            market_name = market.get("marketName", "")
            for entry in market.get("data", []):
                rows.append({
                    "date":       date_str,
                    "stateId":    state_id,
                    "stateName":  state_name,
                    "marketId":   market_id,
                    "marketName": market_name,
                    **entry,
                    "_source":    "ep1_market_report",
                })
    return rows

all_records = []
files = sorted(RAW_DIR.glob("*_endpoint1.json"))
print(f"Found {len(files)} endpoint1 files", flush=True)

for f in files:
    date_str = f.name[:4] + "-" + f.name[4:6] + "-" + f.name[6:8]
    try:
        data = json.loads(f.read_text())
        rows = flatten_endpoint1(data, date_str)
        all_records.extend(rows)
    except Exception as e:
        print(f"  ERROR {f.name}: {e}")

print(f"Total rows: {len(all_records)}", flush=True)

if not all_records:
    print("Nothing to save.")
else:
    all_keys = []
    seen = set()
    for r in all_records:
        for k in r:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    out = "garlic_complete_data.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print(f"Saved to {out}")
    dates = [r["date"] for r in all_records if r.get("date")]
    print(f"Date range: {min(dates)} to {max(dates)}")
    states = set(r["stateName"] for r in all_records if r.get("stateName"))
    print(f"States ({len(states)}): {sorted(states)}")
    markets = set(r["marketName"] for r in all_records if r.get("marketName"))
    print(f"Markets: {len(markets)}")
