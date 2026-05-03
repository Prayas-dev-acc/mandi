#!/usr/bin/env python3
"""
Pull historical weather for Jaora/Mandsaur area from Open-Meteo (free, no key)
and correlate with garlic prices from DuckDB.

Garlic crop calendar (MP):
  Oct–Nov  : Sowing   → rainfall critical for germination
  Nov–Feb  : Growing  → cold nights good, frost bad
  Feb–Apr  : Harvest  → dry preferred, rain causes rot
  May–Sep  : Storage  → heat + humidity = stock loss
"""
import requests, json, duckdb
import pandas as pd
from datetime import date, datetime

# Jaora coordinates
LAT, LON = 23.63, 75.13
LOCATION  = "Jaora, Ratlam District, MP"

DB_PATH = "garlic.duckdb"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────
# 1. PULL WEATHER DATA
# ─────────────────────────────────────────────────────────────────

def fetch_weather(start="2017-01-01", end=None):
    from datetime import timedelta
    if end is None:
        end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    log(f"Fetching weather for {LOCATION} ({start} → {end})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "start_date": start,
        "end_date":   end,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max",
            "et0_fao_evapotranspiration",
            "sunshine_duration",
            "soil_moisture_0_to_7cm_mean",
        ]),
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    daily = data["daily"]

    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.drop(columns=["time"])
    df["year"]  = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month

    log(f"  Got {len(df)} days of weather data")
    df.to_csv("weather_jaora.csv", index=False)
    log(f"  Saved to weather_jaora.csv")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. LOAD GARLIC PRICES
# ─────────────────────────────────────────────────────────────────

JAORA_CLUSTER = [1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727]

def load_prices():
    log("Loading garlic prices from DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(f"""
        SELECT date, ROUND(AVG(modal_price),0) as modal_price,
               ROUND(AVG(arrivals),2) as avg_arrivals
        FROM clean_garlic_prices
        WHERE market_id IN ({','.join(str(x) for x in JAORA_CLUSTER)})
        GROUP BY date ORDER BY date
    """).fetchdf()
    con.close()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    log(f"  Got {len(df)} price days")
    return df


# ─────────────────────────────────────────────────────────────────
# 3. MERGE & DERIVE SEASONAL FEATURES
# ─────────────────────────────────────────────────────────────────

def build_monthly_analysis(weather_df, price_df):
    # Monthly weather aggregates
    w = weather_df.copy()
    w["date_dt"] = pd.to_datetime(w["date"])
    w["ym"] = w["date_dt"].dt.to_period("M")

    wm = w.groupby("ym").agg(
        year=("year", "first"),
        month=("month", "first"),
        avg_max_temp=("temperature_2m_max", "mean"),
        avg_min_temp=("temperature_2m_min", "mean"),
        avg_mean_temp=("temperature_2m_mean", "mean"),
        total_rain_mm=("precipitation_sum", "sum"),
        rainy_days=("precipitation_sum", lambda x: (x > 1).sum()),
        avg_soil_moisture=("soil_moisture_0_to_7cm_mean", "mean"),
        avg_et0=("et0_fao_evapotranspiration", "mean"),
        total_sunshine_hrs=("sunshine_duration", lambda x: x.sum() / 3600),
    ).reset_index()

    # Monthly price aggregates
    p = price_df.copy()
    p["date_dt"] = pd.to_datetime(p["date"])
    p["ym"] = p["date_dt"].dt.to_period("M")

    pm = p.groupby("ym").agg(
        avg_price=("modal_price", "mean"),
        total_arrivals=("avg_arrivals", "sum"),
    ).reset_index()

    merged = wm.merge(pm, on="ym", how="left")
    return merged


def build_seasonal_windows(weather_df):
    """Aggregate weather over crop-relevant windows per year."""
    w = weather_df.copy()
    w["date_dt"] = pd.to_datetime(w["date"])

    rows = []
    for year in range(2017, date.today().year + 1):
        # Sowing window: Oct-Nov of previous year
        sow = w[(w["year"] == year-1) & (w["month"].isin([10, 11]))]
        # Growing window: Dec(prev)-Jan-Feb
        grow_dec = w[(w["year"] == year-1) & (w["month"] == 12)]
        grow_janfeb = w[(w["year"] == year) & (w["month"].isin([1, 2]))]
        grow = pd.concat([grow_dec, grow_janfeb])
        # Harvest window: Mar-Apr
        harv = w[(w["year"] == year) & (w["month"].isin([3, 4]))]
        # Post-harvest/storage: May-Sep
        stor = w[(w["year"] == year) & (w["month"].isin([5, 6, 7, 8, 9]))]

        rows.append({
            "crop_year": year,
            # Sowing
            "sow_rain_mm":      sow["precipitation_sum"].sum() if len(sow) else None,
            "sow_avg_temp":     sow["temperature_2m_mean"].mean() if len(sow) else None,
            # Growing
            "grow_min_temp":    grow["temperature_2m_min"].min() if len(grow) else None,
            "grow_avg_min":     grow["temperature_2m_min"].mean() if len(grow) else None,
            "grow_rain_mm":     grow["precipitation_sum"].sum() if len(grow) else None,
            "frost_days":       (grow["temperature_2m_min"] < 5).sum() if len(grow) else 0,
            # Harvest
            "harv_rain_mm":     harv["precipitation_sum"].sum() if len(harv) else None,
            "harv_avg_temp":    harv["temperature_2m_mean"].mean() if len(harv) else None,
            "harv_rainy_days":  (harv["precipitation_sum"] > 1).sum() if len(harv) else 0,
            # Storage
            "stor_avg_temp":    stor["temperature_2m_mean"].mean() if len(stor) else None,
            "stor_max_temp":    stor["temperature_2m_max"].max() if len(stor) else None,
            "stor_rain_mm":     stor["precipitation_sum"].sum() if len(stor) else None,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# 4. CORRELATE WEATHER WITH PRICES
# ─────────────────────────────────────────────────────────────────

def correlate(seasonal_df, price_df):
    """Join seasonal weather with annual avg price and harvest-month price."""
    con = duckdb.connect(DB_PATH, read_only=True)
    annual = con.execute(f"""
        SELECT YEAR(date) as crop_year,
               ROUND(AVG(modal_price),0) as annual_avg_price,
               ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (3,4)), 0) as harvest_price,
               ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (8,9)), 0) as storage_price,
               ROUND(SUM(arrivals), 0) as total_arrivals
        FROM clean_garlic_prices
        WHERE market_id IN ({','.join(str(x) for x in JAORA_CLUSTER)})
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    con.close()

    merged = seasonal_df.merge(annual, on="crop_year", how="left")
    return merged


# ─────────────────────────────────────────────────────────────────
# 5. PRINT ANALYSIS
# ─────────────────────────────────────────────────────────────────

SEP = "\n" + "═"*72 + "\n"
DIV = "─"*72

def show(title, df):
    print(f"\n{DIV}\n{title}\n{DIV}")
    print(df.to_string(index=False))


def main():
    weather_df = fetch_weather(start="2017-01-01")
    price_df   = load_prices()
    monthly    = build_monthly_analysis(weather_df, price_df)
    seasonal   = build_seasonal_windows(weather_df)
    full       = correlate(seasonal, price_df)

    print(SEP)
    print(f"WEATHER × GARLIC PRICE ANALYSIS — {LOCATION}")
    print(f"Elevation: 469m | Lat {LAT}°N Lon {LON}°E")
    print(SEP)

    # ── A. Sowing Season Weather ──────────────────────────────────
    show("A. SOWING SEASON (Oct–Nov) WEATHER BY YEAR",
         monthly[monthly["month"].isin([10,11])][
             ["year","month","avg_max_temp","avg_min_temp",
              "total_rain_mm","rainy_days","avg_soil_moisture"]
         ].round(1))

    # ── B. Growing Season Cold Analysis ──────────────────────────
    show("B. GROWING SEASON (Dec–Feb) — cold nights & frost risk",
         monthly[monthly["month"].isin([12,1,2])][
             ["year","month","avg_max_temp","avg_min_temp",
              "total_rain_mm","avg_soil_moisture"]
         ].round(1))

    # ── C. Harvest Season Rain ────────────────────────────────────
    show("C. HARVEST SEASON (Mar–Apr) — rain = rot risk",
         monthly[monthly["month"].isin([3,4])][
             ["year","month","total_rain_mm","rainy_days",
              "avg_max_temp","avg_price","total_arrivals"]
         ].round(1))

    # ── D. Storage Season Heat ────────────────────────────────────
    show("D. STORAGE SEASON (May–Sep) — heat & humidity",
         monthly[monthly["month"].isin([5,6,7,8,9])][
             ["year","month","avg_max_temp","avg_min_temp",
              "total_rain_mm","avg_et0","avg_price"]
         ].round(1))

    # ── E. Full Seasonal Summary + Price Correlation ──────────────
    show("E. SEASONAL WEATHER SUMMARY + ANNUAL PRICE (per crop year)",
         full[[
             "crop_year",
             "sow_rain_mm","grow_avg_min","frost_days",
             "harv_rain_mm","harv_rainy_days",
             "stor_avg_temp","stor_rain_mm",
             "harvest_price","storage_price","annual_avg_price","total_arrivals"
         ]].round(1))

    # ── F. Correlation matrix ─────────────────────────────────────
    corr_cols = [
        "sow_rain_mm","grow_avg_min","frost_days",
        "harv_rain_mm","harv_rainy_days",
        "stor_avg_temp","stor_rain_mm",
        "harvest_price","storage_price","annual_avg_price"
    ]
    corr_df = full[corr_cols].dropna()
    if len(corr_df) >= 4:
        print(f"\n{DIV}")
        print("F. CORRELATION MATRIX (weather variables vs prices)")
        print(DIV)
        corr = corr_df.corr()[["harvest_price","storage_price","annual_avg_price"]].round(2)
        print(corr.to_string())

    # ── G. Current sowing season outlook ─────────────────────────
    print(f"\n{DIV}")
    print("G. CURRENT YEAR (2026) WEATHER SNAPSHOT vs HISTORICAL AVG")
    print(DIV)
    this_year = monthly[monthly["year"] == 2026]
    hist_avg  = monthly[monthly["year"] < 2026].groupby("month").agg(
        hist_rain=("total_rain_mm","mean"),
        hist_max_temp=("avg_max_temp","mean"),
        hist_min_temp=("avg_min_temp","mean"),
    ).round(1)
    snap = this_year[["month","total_rain_mm","avg_max_temp","avg_min_temp"]].merge(
        hist_avg, on="month", how="left"
    )
    snap["rain_vs_avg"] = (snap["total_rain_mm"] - snap["hist_rain"]).round(1)
    snap["temp_vs_avg"] = (snap["avg_max_temp"] - snap["hist_max_temp"]).round(1)
    print(snap.to_string(index=False))

    # ── H. Oct-Nov sowing rain vs next year harvest arrivals ──────
    print(f"\n{DIV}")
    print("H. SOWING RAIN (Oct+Nov) → NEXT YEAR ARRIVALS & HARVEST PRICE")
    print("   (more rain at sowing = better germination = more arrivals = lower price)")
    print(DIV)
    sow_rain = monthly[monthly["month"].isin([10,11])].groupby("year")["total_rain_mm"].sum().reset_index()
    sow_rain.columns = ["sow_year","sow_rain_mm"]
    sow_rain["crop_year"] = sow_rain["sow_year"] + 1
    h = full[["crop_year","harvest_price","total_arrivals"]].merge(sow_rain, on="crop_year", how="left")
    print(h[["crop_year","sow_year","sow_rain_mm","total_arrivals","harvest_price"]].round(0).to_string(index=False))

    # Save
    full.to_csv("weather_price_correlation.csv", index=False)
    monthly.to_csv("weather_monthly.csv", index=False)
    log("\nSaved: weather_jaora.csv, weather_monthly.csv, weather_price_correlation.csv")


if __name__ == "__main__":
    main()
