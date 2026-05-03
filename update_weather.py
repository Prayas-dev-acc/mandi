#!/usr/bin/env python3
"""
Update weather_monthly.csv from Open-Meteo (free, no API key).
Fetches from 2017-01-01 to yesterday — Open-Meteo returns full history quickly.
Safe to run daily (overwrites the CSV with fresh data).
"""
import sys
from pathlib import Path

# Import existing functions — avoids duplicating the Open-Meteo call logic
sys.path.insert(0, str(Path(__file__).parent))
from weather_analysis import fetch_weather, build_monthly_analysis, load_prices


def run():
    print("update_weather: fetching from Open-Meteo...", flush=True)
    try:
        weather_df = fetch_weather(start="2017-01-01")
        price_df   = load_prices()
        monthly    = build_monthly_analysis(weather_df, price_df)
        monthly.to_csv("weather_monthly.csv", index=False)
        print(f"update_weather: saved {len(monthly)} months to weather_monthly.csv", flush=True)
        return True
    except Exception as e:
        print(f"update_weather: FAILED — {e}", flush=True)
        return False


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
