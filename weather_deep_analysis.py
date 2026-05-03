#!/usr/bin/env python3
"""
Deep weather × arrivals × price analysis for Jaora garlic cluster.

Core thesis: arrivals are driven by THREE forces:
  1. CROP SIZE   — determined by sowing rain + growing conditions
  2. CROP DAMAGE — rain at harvest causes rot → low arrivals + HIGH price (supply shock)
  3. FARMER BEHAVIOUR — if price is low, farmers hold (cold storage) → low arrivals + low price later

We classify each year by what drove arrivals, then correlate with weather.

Also documents the Jaora "Safed/White Polish" garlic premium.
"""

import pandas as pd
import duckdb
from datetime import datetime

DB = "garlic.duckdb"
CLUSTER = [1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727]
CLUSTER_SQL = ",".join(str(x) for x in CLUSTER)

SEP = "\n" + "═"*74
DIV = "\n" + "─"*74

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)

# ── Load data ────────────────────────────────────────────────────────────────

log("Loading weather...")
wx = pd.read_csv("weather_jaora.csv", parse_dates=["date"])
wx["date"] = wx["date"].dt.date
wx["year"]  = pd.to_datetime(wx["date"]).dt.year
wx["month"] = pd.to_datetime(wx["date"]).dt.month
wx["week"]  = pd.to_datetime(wx["date"]).dt.isocalendar().week.astype(int)

log("Loading prices + arrivals...")
con = duckdb.connect(DB, read_only=True)
daily = con.execute(f"""
    SELECT date,
           ROUND(AVG(modal_price),0)  as price,
           ROUND(SUM(arrivals),2)     as arrivals,
           COUNT(DISTINCT market_id)  as markets_reporting
    FROM clean_garlic_prices
    WHERE market_id IN ({CLUSTER_SQL})
    GROUP BY date ORDER BY date
""").fetchdf()
con.close()
daily["date"]  = pd.to_datetime(daily["date"]).dt.date
daily["year"]  = pd.to_datetime(daily["date"]).dt.year
daily["month"] = pd.to_datetime(daily["date"]).dt.month


# ── PART 1: DAILY RAIN EVENTS → ARRIVAL IMPACT ──────────────────────────────
# For each significant rain event (>10mm/day), measure arrivals in the
# 7-day and 14-day window AFTER the event vs the 7-day window BEFORE.
# This captures the "rain damaged crop → fewer arrivals" effect directly.

log("Analysing rain event impacts...")

def rain_impact_events(wx_df, price_df, min_rain=10, harvest_months=[2,3,4,5]):
    """Find rain events during harvest, measure arrival change after."""
    wx_harv = wx_df[wx_df["month"].isin(harvest_months)].copy()
    heavy   = wx_harv[wx_harv["precipitation_sum"] >= min_rain].copy()

    rows = []
    price_df_idx = price_df.set_index("date")

    for _, ev in heavy.iterrows():
        d = ev["date"]
        rain = ev["precipitation_sum"]
        mo   = ev["month"]
        yr   = ev["year"]

        # 7-day arrivals before and after
        before_dates = pd.date_range(
            end=pd.Timestamp(d) - pd.Timedelta(days=1), periods=7
        ).date
        after_dates  = pd.date_range(
            start=pd.Timestamp(d) + pd.Timedelta(days=1), periods=14
        ).date

        arr_before = price_df_idx.loc[
            price_df_idx.index.isin(before_dates), "arrivals"
        ].mean()
        arr_after  = price_df_idx.loc[
            price_df_idx.index.isin(after_dates), "arrivals"
        ].mean()
        pr_before  = price_df_idx.loc[
            price_df_idx.index.isin(before_dates), "price"
        ].mean()
        pr_after   = price_df_idx.loc[
            price_df_idx.index.isin(after_dates), "price"
        ].mean()

        if pd.notna(arr_before) and pd.notna(arr_after) and arr_before > 0:
            rows.append({
                "date": d,
                "year": yr,
                "month": mo,
                "rain_mm": rain,
                "arr_before_7d": round(arr_before, 1),
                "arr_after_14d": round(arr_after, 1),
                "arr_change_pct": round((arr_after - arr_before) / arr_before * 100, 1),
                "price_before": round(pr_before, 0),
                "price_after": round(pr_after, 0),
                "price_change_pct": round((pr_after - pr_before) / pr_before * 100, 1)
                    if pr_before > 0 else None,
                "supply_shock": arr_after < arr_before * 0.7,  # >30% drop = shock
            })
    return pd.DataFrame(rows)

rain_events = rain_impact_events(wx, daily, min_rain=10)


# ── PART 2: CROP YEAR CLASSIFICATION ────────────────────────────────────────
# Classify each harvest year by what drove the outcome.

log("Classifying crop years...")

def classify_crop_years(wx_df, price_df):
    rows = []
    for year in range(2018, 2027):
        # Weather features
        sow  = wx_df[(wx_df["year"]==year-1) & (wx_df["month"].isin([10,11]))]
        grow = pd.concat([
            wx_df[(wx_df["year"]==year-1) & (wx_df["month"]==12)],
            wx_df[(wx_df["year"]==year) & (wx_df["month"].isin([1,2]))]
        ])
        harv = wx_df[(wx_df["year"]==year) & (wx_df["month"].isin([2,3,4]))]
        stor = wx_df[(wx_df["year"]==year) & (wx_df["month"].isin([5,6,7,8,9]))]

        sow_rain    = sow["precipitation_sum"].sum()
        grow_min    = grow["temperature_2m_min"].mean() if len(grow) else None
        frost_days  = (grow["temperature_2m_min"] < 5).sum() if len(grow) else 0
        harv_rain   = harv["precipitation_sum"].sum()
        harv_heavy  = (harv["precipitation_sum"] >= 10).sum()  # days with >10mm
        stor_temp   = stor["temperature_2m_mean"].mean() if len(stor) else None
        stor_rain   = stor["precipitation_sum"].sum()

        # Price/arrivals features
        yr_p = price_df[price_df["year"] == year]
        harv_p = price_df[(price_df["year"]==year) & (price_df["month"].isin([3,4,5]))]
        stor_p = price_df[(price_df["year"]==year) & (price_df["month"].isin([8,9]))]

        total_arr   = yr_p["arrivals"].sum()
        harv_arr    = harv_p["arrivals"].sum()
        harv_price  = harv_p["price"].mean()
        stor_price  = stor_p["price"].mean()
        annual_price= yr_p["price"].mean()

        # Classify
        prev_year_arr = price_df[price_df["year"]==year-1]["arrivals"].sum()
        arr_vs_prev   = (total_arr / prev_year_arr - 1) * 100 if prev_year_arr > 0 else 0

        # Decision logic
        reasons = []
        stress_score = 0

        if sow_rain < 5:
            reasons.append("DRY SOWING → poor germination → low area planted")
            stress_score += 2
        elif sow_rain > 80:
            reasons.append("WET SOWING → excellent germination → large area")
            stress_score -= 1

        if frost_days > 0:
            reasons.append(f"FROST {int(frost_days)}d → bulb damage → quality/yield loss")
            stress_score += 2

        if grow_min and grow_min < 10.5:
            reasons.append("COLD GROWING SEASON → slower bulb development")
            stress_score += 1

        if harv_heavy >= 3:
            reasons.append(f"HEAVY HARVEST RAIN {int(harv_heavy)}d ≥10mm → field rot → supply shock")
            stress_score += 3
        elif harv_rain > 15:
            reasons.append(f"MODERATE HARVEST RAIN {harv_rain:.0f}mm → quality downgrade")
            stress_score += 1

        if total_arr < 200000 and harv_price and harv_price > 6000:
            reasons.append("LOW ARRIVALS + HIGH HARVEST PRICE → farmer holding in storage")
        elif total_arr < 200000 and harv_price and harv_price < 4000:
            reasons.append("LOW ARRIVALS + LOW PRICE → small crop (weather/area effect)")

        if stor_temp and stor_temp > 30:
            reasons.append("HOT STORAGE SEASON → fast stock depletion → price spike")

        if not reasons:
            reasons.append("NORMAL SEASON — no major stress signals")

        outcome = "BOOM" if annual_price > 8000 else \
                  "GOOD" if annual_price > 5000 else \
                  "AVERAGE" if annual_price > 3000 else "CRASH"

        rows.append({
            "year": year,
            "sow_rain_mm": round(sow_rain, 0),
            "frost_days": int(frost_days),
            "harv_rain_mm": round(harv_rain, 1),
            "harv_heavy_days": int(harv_heavy),
            "stor_avg_temp": round(stor_temp, 1) if stor_temp else None,
            "total_arrivals": round(total_arr, 0),
            "arr_change_pct": round(arr_vs_prev, 0),
            "harvest_price": round(harv_price, 0) if harv_price else None,
            "storage_price": round(stor_price, 0) if stor_price else None,
            "annual_price": round(annual_price, 0) if annual_price else None,
            "outcome": outcome,
            "weather_drivers": " | ".join(reasons),
        })
    return pd.DataFrame(rows)

crop_years = classify_crop_years(wx, daily)


# ── PART 3: PRE-HARVEST RAIN WINDOW ANALYSIS ────────────────────────────────
# Critical window: 30 days before peak arrivals.
# Heavy rain in this window = field damage = low arrivals that year.

log("Analysing pre-harvest stress windows...")

def pre_harvest_rain_analysis(wx_df, price_df):
    """For each year, find rain in Feb 1 – Apr 15 window and its arrival impact."""
    rows = []
    for year in range(2018, 2027):
        # Feb 1 – Apr 15 rain
        start = pd.Timestamp(f"{year}-02-01")
        end   = pd.Timestamp(f"{year}-04-15")
        harv_wx = wx_df[
            (pd.to_datetime(wx_df["date"]) >= start) &
            (pd.to_datetime(wx_df["date"]) <= end)
        ]
        total_rain = harv_wx["precipitation_sum"].sum()
        heavy_days = (harv_wx["precipitation_sum"] >= 10).sum()
        max_daily  = harv_wx["precipitation_sum"].max()

        # Consecutive rainy days (>5mm) — worst for standing crop
        harv_wx_s = harv_wx.sort_values("date")
        rainy = (harv_wx_s["precipitation_sum"] > 5).astype(int)
        max_consec = rainy.groupby((rainy != rainy.shift()).cumsum()).sum().max()

        # Arrivals in Mar–May
        harv_arr = price_df[
            (price_df["year"]==year) & (price_df["month"].isin([3,4,5]))
        ]["arrivals"].sum()
        harv_price = price_df[
            (price_df["year"]==year) & (price_df["month"].isin([3,4,5]))
        ]["price"].mean()

        rows.append({
            "year": year,
            "feb_apr_rain_mm": round(total_rain, 1),
            "heavy_rain_days": int(heavy_days),
            "max_daily_rain": round(max_daily, 1),
            "max_consec_rainy_days": int(max_consec) if pd.notna(max_consec) else 0,
            "mar_may_arrivals": round(harv_arr, 0),
            "mar_may_avg_price": round(harv_price, 0) if pd.notna(harv_price) else None,
        })
    return pd.DataFrame(rows)

pre_harv = pre_harvest_rain_analysis(wx, daily)


# ── PART 4: WEEKLY ARRIVAL DROPS AFTER RAIN ──────────────────────────────────

log("Finding weekly supply shocks...")

def weekly_shock_analysis(wx_df, price_df):
    """Rolling weekly arrivals — find weeks with >30% drop after rain events."""
    merged = pd.merge(
        price_df, wx_df[["date","precipitation_sum","temperature_2m_max"]],
        on="date", how="left"
    )
    merged = merged.sort_values("date")
    merged["rolling_arr_7d"]  = merged["arrivals"].rolling(7, min_periods=3).mean()
    merged["rolling_rain_7d"] = merged["precipitation_sum"].rolling(7, min_periods=5).sum()
    merged["arr_drop_7d"]     = merged["rolling_arr_7d"].pct_change(7) * 100

    # Supply shocks: arrivals drop >35% and there was rain in prior 7 days
    shocks = merged[
        (merged["month"].isin([2,3,4,5])) &
        (merged["arr_drop_7d"] < -35) &
        (merged["rolling_rain_7d"] > 15) &
        (merged["rolling_arr_7d"] > 20)   # only when market was active
    ][["date","year","month","price","arrivals","rolling_rain_7d","arr_drop_7d"]].copy()
    shocks = shocks.rename(columns={
        "rolling_rain_7d": "rain_prior_7d_mm",
        "arr_drop_7d": "arrival_drop_pct"
    })
    return shocks.round(1)

shocks = weekly_shock_analysis(wx, daily)


# ── PRINT ALL RESULTS ─────────────────────────────────────────────────────────

print(SEP)
print("DEEP WEATHER × ARRIVALS × PRICE ANALYSIS — JAORA GARLIC CLUSTER")
print(SEP)

print(DIV)
print("PART 1: SIGNIFICANT RAIN EVENTS (≥10mm) DURING HARVEST (Feb–May)")
print("       and their measured impact on arrivals in following 14 days")
print(DIV)
if len(rain_events):
    print(rain_events[[
        "date","year","month","rain_mm",
        "arr_before_7d","arr_after_14d","arr_change_pct",
        "price_before","price_after","price_change_pct","supply_shock"
    ]].sort_values("date").to_string(index=False))
else:
    print("No significant events found.")

print(DIV)
print("PART 2: CROP YEAR CLASSIFICATION — what drove each year's outcome")
print(DIV)
for _, row in crop_years.iterrows():
    print(f"\n  {int(row.year)} [{row.outcome}] — Annual avg ₹{row.annual_price:,.0f} | "
          f"Arrivals {row.total_arrivals:,.0f} MT ({row.arr_change_pct:+.0f}% vs prior yr)")
    print(f"  Harvest ₹{row.harvest_price:,.0f} → Storage ₹{row.storage_price if pd.notna(row.storage_price) else 'N/A'}")
    print(f"  Weather: sow_rain={row.sow_rain_mm}mm | frost={row.frost_days}d | "
          f"harv_rain={row.harv_rain_mm}mm ({row.harv_heavy_days} heavy days) | "
          f"stor_temp={row.stor_avg_temp}°C")
    print(f"  DRIVERS: {row.weather_drivers}")

print(DIV)
print("PART 3: PRE-HARVEST STRESS WINDOW (Feb 1 – Apr 15) vs ARRIVALS")
print("        Heavy rain in this window = standing crop damage = supply shock")
print(DIV)
print(pre_harv.to_string(index=False))

print(DIV)
print("PART 4: WEEKLY SUPPLY SHOCKS — rain events that caused >35% arrival drops")
print(DIV)
if len(shocks):
    print(shocks.sort_values("date").to_string(index=False))
else:
    print("No clear weekly shocks found with current thresholds.")

print(DIV)
print("PART 5: CORRELATION — pre-harvest rain vs harvest arrivals & price")
print(DIV)
corr_df = pre_harv[["feb_apr_rain_mm","heavy_rain_days","max_consec_rainy_days",
                      "mar_may_arrivals","mar_may_avg_price"]].dropna()
if len(corr_df) >= 4:
    c = corr_df.corr()[["mar_may_arrivals","mar_may_avg_price"]].round(2)
    print(c.to_string())
    print("\n  Interpretation:")
    print("  Negative corr(rain, arrivals) = more rain → fewer arrivals (crop damage)")
    print("  Positive corr(rain, price)    = more rain → higher price (supply shock)")

print(DIV)
print("PART 6: JAORA 'SAFED / WHITE POLISH' GARLIC PREMIUM")
print("        Why Jaora consistently prices above MP state average")
print(DIV)

con2 = duckdb.connect(DB, read_only=True)

print("\n6a. Variety breakdown at Jaora vs rest of MP:")
print(con2.execute("""
    SELECT
        CASE WHEN market_id = 1085 THEN 'Jaora' ELSE 'Other MP' END as mkt,
        variety, grade,
        COUNT(*) as days,
        ROUND(AVG(modal_price),0) as avg_price,
        ROUND(AVG(arrivals),1) as avg_daily_arrivals_mt
    FROM clean_garlic_prices
    WHERE state_name = 'Madhya Pradesh'
    GROUP BY 1,2,3
    HAVING COUNT(*) > 50
    ORDER BY 1, avg_price DESC
""").fetchdf().to_string(index=False))

print("\n6b. Jaora 'Garlic' variety (Safed/White) premium over 'Desi' by year:")
print(con2.execute("""
    SELECT YEAR(date) as year,
        ROUND(AVG(modal_price) FILTER (WHERE variety='Garlic'), 0) as safed_price,
        ROUND(AVG(modal_price) FILTER (WHERE variety='Desi'),   0) as desi_price,
        ROUND(AVG(modal_price) FILTER (WHERE variety='Garlic') -
              AVG(modal_price) FILTER (WHERE variety='Desi'), 0)   as premium,
        ROUND(
          (AVG(modal_price) FILTER (WHERE variety='Garlic') /
           NULLIF(AVG(modal_price) FILTER (WHERE variety='Desi'),0) - 1) * 100
        , 1) as premium_pct
    FROM clean_garlic_prices
    WHERE market_id = 1085
    GROUP BY 1 ORDER BY 1
""").fetchdf().to_string(index=False))

print("\n6c. Jaora Safed price vs national benchmark (top 5 states):")
print(con2.execute("""
    SELECT state_name,
           ROUND(AVG(modal_price) FILTER (WHERE variety IN ('Garlic','Average')), 0) as white_garlic_price,
           ROUND(SUM(arrivals), 0) as total_arrivals_mt
    FROM clean_garlic_prices
    WHERE variety IN ('Garlic','Average')
    GROUP BY 1
    HAVING SUM(arrivals) > 10000
    ORDER BY 2 DESC LIMIT 10
""").fetchdf().to_string(index=False))

print("\n6d. Jaora Organic premium (emerging opportunity):")
print(con2.execute("""
    SELECT YEAR(date) as year,
        ROUND(AVG(modal_price) FILTER (WHERE variety='Garlic-Organic'), 0) as organic_price,
        ROUND(AVG(modal_price) FILTER (WHERE variety='Garlic'),         0) as standard_price,
        ROUND(AVG(modal_price) FILTER (WHERE variety='Garlic-Organic') -
              AVG(modal_price) FILTER (WHERE variety='Garlic'), 0)         as organic_premium,
        COUNT(*) FILTER (WHERE variety='Garlic-Organic') as organic_days
    FROM clean_garlic_prices
    WHERE market_id = 1085
    GROUP BY 1 HAVING organic_days > 0 ORDER BY 1
""").fetchdf().to_string(index=False))

con2.close()

# Save
rain_events.to_csv("rain_arrival_events.csv", index=False)
crop_years.to_csv("crop_year_classification.csv", index=False)
pre_harv.to_csv("pre_harvest_stress.csv", index=False)
log("Saved: rain_arrival_events.csv, crop_year_classification.csv, pre_harvest_stress.csv")
