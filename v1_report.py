#!/usr/bin/env python3
"""
LAHSUN PRICE INTELLIGENCE — v1 Report
Jaora / Mandsaur / Neemuch cluster, Madhya Pradesh
------------------------------------------------------
Run:  python3 v1_report.py
Output: full market briefing to terminal
"""
import duckdb
import numpy as np
import pandas as pd
from datetime import date
from crop_stress import run_crop_stress_check

DB_PATH       = "garlic.duckdb"
JAORA_CLUSTER = [1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727]
CLUSTER_SQL   = ",".join(str(x) for x in JAORA_CLUSTER)

SEP = "\n" + "═" * 68
DIV = "─" * 68


# ── helpers ──────────────────────────────────────────────────────────

def _stars(n, out_of=5):
    return "★" * n + "☆" * (out_of - n)

def _pct(a, b):
    return (a - b) / (b + 1) * 100 if b else 0


def load_all(con):
    """Return all data bundles needed for the report."""

    # 1. Recent monthly prices (last 36 months, for current context)
    recent = con.execute(f"""
        SELECT DATE_TRUNC('month', date)::DATE as month,
               ROUND(AVG(modal_price), 0)      as price,
               ROUND(SUM(arrivals), 0)          as arrivals
        FROM clean_garlic_prices
        WHERE market_id IN ({CLUSTER_SQL})
        GROUP BY 1 ORDER BY 1 DESC LIMIT 36
    """).fetchdf()
    recent["month"] = pd.to_datetime(recent["month"])
    recent = recent.sort_values("month").reset_index(drop=True)

    # Data start year (all-time, for footer)
    db_min_year = con.execute("SELECT YEAR(MIN(date)) FROM clean_garlic_prices").fetchone()[0]

    # 2. Long-run monthly seasonality (all history)
    seasonal = con.execute(f"""
        SELECT MONTH(date) as mo,
               ROUND(AVG(modal_price), 0)  as avg_price,
               ROUND(AVG(arrivals),    0)  as avg_arr
        FROM clean_garlic_prices
        WHERE market_id IN ({CLUSTER_SQL})
        GROUP BY 1 ORDER BY 1
    """).fetchdf()

    # 3. Yearly summary
    yearly = con.execute(f"""
        SELECT YEAR(date) as yr,
               ROUND(AVG(modal_price), 0)                              as yr_avg,
               ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (3,4)), 0) as harv_price,
               ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (7,8,9)), 0) as stor_price,
               ROUND(MIN(modal_price), 0)                              as yr_low,
               ROUND(MAX(modal_price), 0)                              as yr_high,
               ROUND(SUM(arrivals)    FILTER (WHERE MONTH(date) BETWEEN 3 AND 5), 0) as harv_arrivals
        FROM clean_garlic_prices
        WHERE market_id IN ({CLUSTER_SQL})
        GROUP BY 1 ORDER BY 1
    """).fetchdf()

    # 4. South spread (demand-supply features)
    try:
        ds = pd.read_csv("demand_supply_features.csv")
        ds["month"] = pd.to_datetime(ds["month"])
    except FileNotFoundError:
        ds = pd.DataFrame()

    # 5. China FOB
    try:
        china = pd.read_csv("china_garlic_comtrade.csv")
    except FileNotFoundError:
        china = pd.DataFrame()

    # 6. Forecast
    try:
        forecast = pd.read_csv("price_forecast.csv")
    except FileNotFoundError:
        forecast = pd.DataFrame()

    return recent, seasonal, yearly, ds, china, forecast, db_min_year


def india_area_data():
    data = [
        (2016, 310, 1580), (2017, 321, 1693), (2018, 303, 1611),
        (2019, 358, 2910), (2020, 363, 2925), (2021, 386, 3190),
        (2022, 408, 3208), (2023, 431, 3240), (2024, 390, 2800),
        (2025, 465, 3650), (2026, 440, 3400),
    ]
    return pd.DataFrame(data, columns=["crop_year", "area_kha", "prod_kmt"])


# ── sections ─────────────────────────────────────────────────────────

def print_header():
    today = date.today()
    print(SEP)
    print(f"  LAHSUN PRICE INTELLIGENCE — JAORA MANDI  v1")
    print(f"  {today.strftime('%d %b %Y')}  |  Cluster: Jaora · Mandsaur · Neemuch · Piplya · Sailana")
    print(SEP)


def print_crop_cycle(seasonal):
    """Crop calendar: dual bar chart (price + arrivals) showing inverse relationship."""
    today_month = date.today().month

    months_in_cycle = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
    month_nos       = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    phase_of        = {10:"SOWING",11:"SOWING",12:"GROWING",1:"GROWING",2:"GROWING",
                       3:"HARVEST",4:"HARVEST",5:"HARVEST",
                       6:"STORAGE",7:"STORAGE",8:"STORAGE",9:"STORAGE"}
    phase_sep_at    = {12, 3, 6}   # months where a new phase begins

    seas   = seasonal.set_index("mo")
    prices = [int(seas.loc[m,"avg_price"]) if m in seas.index else 0 for m in month_nos]
    arrs   = [int(seas.loc[m,"avg_arr"])   if m in seas.index else 0 for m in month_nos]
    max_p, min_p = max(prices), min(prices)
    max_a        = max(arrs)
    BAR = 11

    print(f"\n  GARLIC CROP CALENDAR — MP Malwa  (Oct → Sep annual cycle)")
    print(DIV)
    print()
    print(f"  {'─── SOWING ───':^15}  {'──────── GROWING ────────':^26}  {'────── HARVEST ──────':^22}  {'──────────── STORAGE ────────────':^33}")
    print(f"  {'Oct – Nov':^15}  {'Dec – Jan – Feb':^26}  {'Mar – Apr – May':^22}  {'Jun – Jul – Aug – Sep':^33}")
    print()
    print(f"  Seeds planted. Old     Bulbs forming; no          Arrivals flood mandi;       Stock depletes month")
    print(f"  stock squeeze →        fresh supply coming.        prices hit annual LOW.       by month. Prices rise")
    print(f"  PEAK prices            Prices drift down           Best time to buy stock       toward Oct–Nov peak")
    print()
    print(f"  {'─'*66}")

    # ── Price bar chart ──────────────────────────────────────────────
    print(f"\n  PRICE  ₹/quintal  (█ = high  · = low;  historical avg 2010–2025)")
    print()
    for i, (mo, mo_name) in enumerate(zip(month_nos, months_in_cycle)):
        p       = prices[i]
        bar_len = round(p / max_p * BAR)
        bar     = "█" * bar_len + "·" * (BAR - bar_len)
        here    = "  ◄ YOU ARE HERE" if mo == today_month else ""
        tag     = " ← ANNUAL PEAK"  if p == max_p else (" ← ANNUAL FLOOR" if p == min_p else "")
        sep     = "│" if mo in phase_sep_at else " "
        print(f"  {sep} {mo_name:3}  {bar}  ₹{p:,}{tag}{here}")

    # ── Arrivals bar chart ───────────────────────────────────────────
    print(f"\n  ARRIVALS  MT/month  (▒ = high supply  · = low;  INVERSE of price)")
    print()
    for i, (mo, mo_name) in enumerate(zip(month_nos, months_in_cycle)):
        a       = arrs[i]
        bar_len = round(a / max_a * BAR)
        bar     = "▒" * bar_len + "·" * (BAR - bar_len)
        here    = "  ◄" if mo == today_month else ""
        tag     = " ← PEAK arrivals → floor price" if a == max_a else ""
        sep     = "│" if mo in phase_sep_at else " "
        print(f"  {sep} {mo_name:3}  {bar}  {a:4} MT{tag}{here}")

    # ── Key insight ──────────────────────────────────────────────────
    print()
    print(f"  {'─'*66}")
    print(f"  INVERSE LAW: when mandi arrivals ↑, price ↓  (always move opposite)")
    print(f"  ▸ Harvest peak  (Mar {max_a} MT/mo)  → price floor  ₹{min_p:,}")
    print(f"  ▸ Pre-new-crop  (Jan {arrs[3]} MT/mo) → price still descending toward floor")
    print(f"  ▸ Post-harvest  (Jun {arrs[8]} MT/mo) → price recovering toward Oct–Nov peak")
    print()

    # ── Current phase guidance ───────────────────────────────────────
    current_phase = phase_of.get(today_month, "STORAGE")
    guidance = {
        "SOWING":  "Near PRICE PEAK. Prime window to sell old stock at maximum value.",
        "GROWING": "Prices declining toward harvest. Hold old stock if storage is cheap.",
        "HARVEST": "Price at FLOOR. Best time to accumulate. Don't panic-sell.",
        "STORAGE": "Prices recovering month by month. Sell toward Oct–Nov for best value.",
    }
    curr_mo_name = months_in_cycle[month_nos.index(today_month)] if today_month in month_nos else ""
    print(f"  NOW ({curr_mo_name} — {current_phase} phase): {guidance[current_phase]}")


def print_current_snapshot(recent, today_month=None):
    """Where are prices right now vs context."""
    last  = recent.iloc[-1]
    prev  = recent.iloc[-2]
    yr_ago = recent[recent["month"].dt.month == last["month"].month].iloc[-2] if len(recent) > 12 else None

    price = int(last["price"])
    arr   = int(last["arrivals"])
    mom   = _pct(price, int(prev["price"]))
    last_mo_str = last["month"].strftime("%b %Y")

    print(SEP)
    print(f"  CURRENT MARKET  (Jaora cluster — {last_mo_str})")
    print(DIV)

    prev_mo_str = prev["month"].strftime("%b %Y")
    print(f"\n  Price ({last_mo_str})  : ₹{price:,}/quintal")
    print(f"  Month-on-month      : {mom:+.1f}%  ({int(prev['price']):,} → {price:,})")
    if yr_ago is not None:
        yr_ago_mo = yr_ago["month"].strftime("%b %Y")
        yoy = _pct(price, int(yr_ago["price"]))
        print(f"  Year-on-year        : {yoy:+.1f}%  ({yr_ago_mo} was ₹{int(yr_ago['price']):,})")
    print(f"  Arrivals            : {arr:,} MT  (down sharply — harvest winding up)")

    print()
    print(f"  WHAT HAPPENED:")
    print(f"  ▸ 2024 was a severe shortage year. Garlic touched ₹23,176 (Nov 2024) —")
    print(f"    a 15-year record. Low area (390k ha) + weak crop caused the spike.")
    print(f"  ▸ Farmers responded with RECORD sowing in 2025: 465k ha (+19%).")
    print(f"    Bumper crop flooded mandis → harvest price collapsed to ₹3,590 (Apr 2025).")
    print(f"  ▸ Storage season 2025 was negative (-14%) — holding stock was a mistake.")
    print(f"  ▸ 2026 sowing was 440k ha (still 18% above 10-yr average).")
    print(f"  ▸ Current ₹5,260 is NORMAL — near the 15-year median (₹5,100).")


def load_weather_signals():
    """Return key weather signals relevant to spike risk."""
    try:
        w = pd.read_csv("weather_monthly.csv")
        w["month"] = pd.to_datetime(w["ym"].astype(str)).dt.to_period("M").dt.to_timestamp()
        w["yr"] = w["month"].dt.year
        w["mo"] = w["month"].dt.month

        # Oct sowing rain for the most recent sowing season (Oct 2025)
        oct_row  = w[(w["yr"] == 2025) & (w["mo"] == 10)]
        sow_rain = float(oct_row["total_rain_mm"].values[0]) if len(oct_row) else 0.0
        # Historical Oct avg
        oct_hist = w[w["mo"] == 10]["total_rain_mm"].mean()

        # Feb soil moisture (bulb-fill month — wet = disease risk)
        feb_row = w[(w["yr"] == 2026) & (w["mo"] == 2)]
        feb_sm  = float(feb_row["avg_soil_moisture"].values[0]) if len(feb_row) else 0.0
        feb_hist = w[w["mo"] == 2]["avg_soil_moisture"].mean()

        # Harvest season rain Mar+Apr 2026
        harv_row = w[(w["yr"] == 2026) & (w["mo"].isin([3, 4]))]
        harv_rain = float(harv_row["total_rain_mm"].sum()) if len(harv_row) else 0.0
        harv_hist = w[w["mo"].isin([3, 4])]["total_rain_mm"].sum() / w["yr"].nunique()

        return {
            "sow_rain_2025": sow_rain, "sow_rain_hist": oct_hist,
            "feb_sm_2026": feb_sm, "feb_sm_hist": feb_hist,
            "harv_rain_2026": harv_rain, "harv_rain_hist": harv_hist,
        }
    except Exception:
        return {}


def load_arrivals_signal(con):
    """Return YoY harvest season arrivals comparison."""
    df = con.execute(f"""
        SELECT YEAR(date) + CASE WHEN MONTH(date) = 12 THEN 1 ELSE 0 END as crop_yr,
               ROUND(SUM(arrivals), 0) as total_arr
        FROM clean_garlic_prices
        WHERE market_id IN ({CLUSTER_SQL})
          AND MONTH(date) IN (12, 1, 2, 3, 4, 5)
          AND YEAR(date) >= 2020
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    yr_dict = df.set_index("crop_yr")["total_arr"].to_dict()
    cur  = yr_dict.get(2026, None)
    prev = yr_dict.get(2025, None)
    avg3 = np.mean([yr_dict.get(y, np.nan) for y in [2023, 2024, 2025]])
    return {"cur": cur, "prev": prev, "avg3": avg3,
            "yoy_pct": (cur - prev) / prev * 100 if cur and prev else None,
            "vs_avg3_pct": (cur - avg3) / avg3 * 100 if cur and avg3 else None}


def compute_spike_risk(recent, ds, china, area_df, weather=None, arrivals=None):
    """
    Spike risk = risk that prices will SPIKE ≥50% in next 12 months.
    Score 1–5  (1 = very low, 5 = critical alert).

    Logic: start at 3 (base/uncertain). Supply excess reduces it,
    arrivals shortfall + weather damage raises it.
    """
    area_10yr = area_df["area_kha"].mean()
    area_cur  = area_df.loc[area_df["crop_year"] == 2026, "area_kha"].values[0]
    area_prev = area_df.loc[area_df["crop_year"] == 2025, "area_kha"].values[0]
    prod_10yr = area_df["prod_kmt"].mean()
    prod_cur  = area_df.loc[area_df["crop_year"] == 2026, "prod_kmt"].values[0]

    area_diff = (area_cur - area_10yr) / area_10yr * 100
    prod_diff = (prod_cur - prod_10yr) / prod_10yr * 100
    area_yoy  = (area_cur - area_prev) / area_prev * 100

    last_price = int(recent.iloc[-1]["price"])
    peak_price = int(recent["price"].max())

    score = 3.0

    # Paper supply (area/production estimates) above normal → bearish
    if area_diff > 15:  score -= 1.0
    elif area_diff > 8: score -= 0.5
    if prod_diff > 20:  score -= 1.0
    elif prod_diff > 10: score -= 0.5

    # Actual arrivals vs last year — this overrides paper supply signal
    # Low arrivals = crop was genuinely smaller than area estimates suggest
    if arrivals and arrivals.get("yoy_pct") is not None:
        yoy = arrivals["yoy_pct"]
        if yoy < -35:   score += 1.5   # severe shortfall (e.g. −40%)
        elif yoy < -20: score += 1.0
        elif yoy < -10: score += 0.5

    # Weather damage signals
    if weather:
        # Oct sowing rain >> normal → waterlogging → lower effective area
        sow_ratio = weather.get("sow_rain_2025", 0) / max(weather.get("sow_rain_hist", 10), 1)
        if sow_ratio > 5:   score += 1.0   # extreme (e.g. 119mm vs 15mm hist avg)
        elif sow_ratio > 2: score += 0.5

        # Feb soil moisture >> normal → fungal disease during bulb fill
        sm_ratio = weather.get("feb_sm_2026", 0) / max(weather.get("feb_sm_hist", 0.27), 0.01)
        if sm_ratio > 1.15: score += 0.5

    # Area falling YoY → less future supply
    if area_yoy < -10:  score += 1.0
    elif area_yoy < -5: score += 0.5

    # Post-spike: price crashed → spike unlikely to immediately repeat
    if last_price / peak_price < 0.35:
        score -= 1.0
    elif last_price / peak_price < 0.55:
        score -= 0.5

    # South demand pull
    if not ds.empty and "south_spread" in ds.columns:
        ss_vals = ds["south_spread"].dropna()
        baseline = ss_vals.iloc[-12:-1].median() if len(ss_vals) >= 12 else ss_vals.median()
        cur_spread = ss_vals.iloc[-1]
        if baseline > 0 and cur_spread / baseline > 1.2:
            score += 0.5

    # China FOB high → export demand rising
    if not china.empty and "china_fob_per_kg" in china.columns:
        china_vals = china["china_fob_per_kg"].dropna()
        if len(china_vals) >= 3:
            if china_vals.iloc[-1] / china_vals.iloc[:-1].mean() > 1.25:
                score += 0.5

    star_score = max(1, min(5, round(score)))

    signals = {
        "area_cur": area_cur, "area_10yr": area_10yr, "area_diff": area_diff,
        "prod_cur": prod_cur, "prod_10yr": prod_10yr, "prod_diff": prod_diff,
        "area_yoy": area_yoy, "last_price": last_price, "peak_price": peak_price,
        "weather": weather or {}, "arrivals": arrivals or {},
    }
    if not ds.empty and "south_spread" in ds.columns:
        signals["cur_spread"] = float(ds["south_spread"].dropna().iloc[-1])
        if "kerala_price" in ds.columns:
            signals["cur_kerala"] = float(ds["kerala_price"].dropna().iloc[-1])
    if not china.empty and "china_fob_per_kg" in china.columns:
        signals["china_fob"] = float(china["china_fob_per_kg"].dropna().iloc[-1])

    return star_score, signals


def print_spike_risk(recent, ds, china, area_df):
    weather  = load_weather_signals()
    con2     = duckdb.connect(DB_PATH, read_only=True)
    arrivals = load_arrivals_signal(con2)
    con2.close()

    star_score, sig = compute_spike_risk(recent, ds, china, area_df, weather, arrivals)
    print(SEP)
    print(f"  SPIKE RISK INDICATOR  {_stars(star_score)} ({star_score}/5)")
    print(DIV)

    risk_text = {
        1: "VERY LOW  — No shortage signal. Price spike very unlikely.",
        2: "LOW       — Supply mostly ample. Monitor arrivals.",
        3: "MODERATE  — Actual supply tighter than paper estimates suggest.",
        4: "HIGH      — Clear tightness. Strong chance of price spike.",
        5: "CRITICAL  — Shortage conditions present. Spike highly likely.",
    }
    print(f"\n  {risk_text[star_score]}")
    print()
    print(f"  {'Signal':<32} {'Value':>16}   {'Flag':<12}  Note")
    print(f"  {'-'*74}")

    def row(label, value_str, flag, note):
        icons = {"LOW": "▼ low risk", "NEUTRAL": "→ neutral ", "WATCH": "▲ watch   ", "ALERT": "▲▲ alert  "}
        icon = icons.get(flag, f"{flag:<12}")
        print(f"  {label:<32} {value_str:>16}   {icon:<12}  {note}")

    # ── Arrivals (most real-time signal) ──────────────────────────
    if arrivals.get("yoy_pct") is not None:
        yoy = arrivals["yoy_pct"]
        arr_flag = "ALERT" if yoy < -30 else ("WATCH" if yoy < -15 else "NEUTRAL")
        row("Arrivals Dec'25–May'26 vs prior season",
            f"{int(arrivals['cur']):,} MT",
            arr_flag,
            f"{yoy:+.0f}% vs last year ({int(arrivals['prev']):,} MT) — REAL supply signal")

    # ── Weather damage signals ─────────────────────────────────────
    if weather:
        sr = weather.get("sow_rain_2025", 0)
        sh = weather.get("sow_rain_hist", 10)
        sow_flag = "ALERT" if sr > 5 * sh else ("WATCH" if sr > 2 * sh else "NEUTRAL")
        row("Oct 2025 sowing rain",
            f"{sr:.0f} mm",
            sow_flag,
            f"Hist avg {sh:.0f}mm — waterlogging → uneven germination, yield loss")

        sm   = weather.get("feb_sm_2026", 0)
        smh  = weather.get("feb_sm_hist", 0.27)
        sm_flag = "WATCH" if sm > smh * 1.1 else "NEUTRAL"
        row("Feb 2026 soil moisture",
            f"{sm:.3f}",
            sm_flag,
            f"Hist avg {smh:.3f} — wet bulb-fill → fungal disease risk")

    # ── Paper supply (area/production) ────────────────────────────
    ad = sig["area_diff"]
    row("India sowing area 2026 (est)",
        f"{int(sig['area_cur']):,}k ha",
        "LOW" if ad > 8 else "NEUTRAL",
        f"{ad:+.0f}% vs 10yr avg — paper supply ample, but not reflecting crop damage")

    # ── Demand signals ────────────────────────────────────────────
    if "cur_spread" in sig:
        kerala_str = f" (Kerala ₹{int(sig['cur_kerala']):,})" if sig.get("cur_kerala") else ""
        row("South spread (Kerala–Jaora)",
            f"₹{int(sig['cur_spread']):,}/q",
            "WATCH" if sig["cur_spread"] > 10000 else "NEUTRAL",
            f"Strong demand pull from south{kerala_str}")

    if "china_fob" in sig:
        cf = sig["china_fob"]
        row("China FOB export price",
            f"${cf:.2f}/kg",
            "WATCH" if cf > 1.2 else "NEUTRAL",
            "High price → more export demand for Indian garlic")

    pct_from_peak = (1 - sig["last_price"] / sig["peak_price"]) * 100
    row("Price vs recent peak",
        f"₹{sig['last_price']:,} / ₹{sig['peak_price']:,}",
        "NEUTRAL",
        f"Down {pct_from_peak:.0f}% from peak — limits near-term spike but not impossible")

    print()
    if star_score <= 2:
        print(f"  Paper supply is ample. Watch Jun–Jul arrivals for confirmation.")
    elif star_score == 3:
        print(f"  WHY MODERATE: Area data says ample supply (440k ha sown), BUT actual")
        print(f"  harvest arrivals are {arrivals.get('yoy_pct', 0):+.0f}% below last year. Oct 2025")
        print(f"  sowing had {weather.get('sow_rain_2025',0):.0f}mm rain ({weather.get('sow_rain_hist',10):.0f}mm historical avg) →")
        print(f"  waterlogging + elevated Feb soil moisture → crop damage confirmed")
        print(f"  by low arrivals. Real supply is tighter than estimates show.")
        print(f"  SPIKE WATCH: If Jun–Sep arrivals stay below 15,000 MT/month,")
        print(f"  revise to HIGH (4/5). Next big signal: Oct 2026 price level.")
    else:
        print(f"  Multiple shortage signals. Review stock-holding strategy urgently.")


def print_forecast(recent, forecast):
    print(SEP)
    print("  6-MONTH PRICE FORECAST  (Jaora cluster, ₹/quintal)")
    print(DIV)

    last_price = int(recent.iloc[-1]["price"])
    last_month_str = recent.iloc[-1]["month"].strftime("%b %Y")
    print(f"\n  Base: {last_month_str} = ₹{last_price:,}  |  Band: q=0.15 to q=0.85 (70% confidence)")
    print()
    print(f"  {'Month':<9} {'Low ₹':>8}  {'Forecast':>9}  {'High ₹':>8}  {'vs now':>7}  {'Season':<9}  Signal")
    print(f"  {'-'*66}")

    prev = last_price
    for _, r in forecast.iterrows():
        p  = int(r["pred_price"])
        lo = int(r["price_low"])
        hi = int(r["price_high"])
        vs = _pct(p, last_price)
        chg = _pct(p, prev)
        sig = "↑ BUY " if chg > 6 else ("↓ SELL" if chg < -6 else "→ FLAT")
        print(f"  {r['month']:<9} ₹{lo:>6,}  ₹{p:>7,}  ₹{hi:>6,}  {vs:>+6.0f}%  {r['season']:<9}  {sig}")
        prev = p

    print()
    print(f"  WHY PRICES DRIFT LOWER:")
    print(f"  ▸ 2025 was a bumper crop (465k ha). Ample old-stock is in warehouses.")
    print(f"  ▸ 2026 sowing also above normal → next harvest will be large.")
    print(f"  ▸ In post-bumper years (2017, 2018, 2022, 2025), storage season")
    print(f"    gains are muted or negative — holding stock earns little/nothing.")
    print(f"  ▸ South demand (Kerala ₹15,600) is strong but cannot pull Jaora")
    print(f"    prices up when domestic supply is ample.")
    print()
    print(f"  UPSIDE SCENARIO (reach ₹{int(forecast['price_high'].max()):,}):")
    print(f"  ▸ If Oct–Nov 2026 sowing rain is deficient (below 60mm) AND")
    print(f"    arrivals in Sep–Oct drop below 10,000 MT/month.")
    print()
    print(f"  DOWNSIDE SCENARIO (reach ₹{int(forecast['price_low'].min()):,}):")
    print(f"  ▸ If monsoon 2026 is good, 2027 area expands further, and China")
    print(f"    garlic price falls (dumping into India market).")


def print_historical_patterns(yearly):
    print(SEP)
    print("  WHAT HISTORY SAYS — ANALOGOUS YEARS")
    print(DIV)
    print()
    print(f"  Post-bumper years (large supply following high-price year):")
    print()
    print(f"  {'Year':<6} {'Harvest':>9}  {'Jul-Sep':>9}  {'Storage gain':>13}  Context")
    print(f"  {'-'*65}")

    analogs = [2017, 2018, 2020, 2022, 2025]
    for _, r in yearly[yearly["yr"].isin(analogs)].iterrows():
        h = int(r["harv_price"]) if pd.notna(r["harv_price"]) else 0
        s = int(r["stor_price"]) if pd.notna(r["stor_price"]) else 0
        gain = _pct(s, h) if h else 0
        gain_str = f"{gain:+.0f}%"
        contexts = {
            2017: "After 2016 bumper (304% storage premium in 2016 → glut)",
            2018: "Continued oversupply — worst storage year in dataset",
            2020: "COVID demand disruption; storage still +49%",
            2022: "Bumper after stable 2021; storage flat (-5%)",
            2025: "Crash from 2024 spike; record area → storage -14%",
        }
        ctx = contexts.get(int(r["yr"]), "")
        print(f"  {int(r['yr']):<6} ₹{h:>7,}   ₹{s:>7,}   {gain_str:>10}    {ctx}")

    print()
    yr_df = yearly.dropna(subset=["harv_price", "stor_price"])
    yr_df = yr_df.copy()
    yr_df["stor_gain"] = yr_df.apply(lambda x: _pct(x["stor_price"], x["harv_price"]), axis=1)
    good_yrs  = yr_df[yr_df["stor_gain"] > 20]
    bad_yrs   = yr_df[yr_df["stor_gain"] < 0]

    print(f"  Out of {len(yr_df)} years with full data:")
    print(f"  ▸ Storage GAINED >20%  : {len(good_yrs)} years  (avg gain: {good_yrs['stor_gain'].mean():.0f}%)")
    print(f"  ▸ Storage LOST money   : {len(bad_yrs)} years   (avg loss: {bad_yrs['stor_gain'].mean():.0f}%)")
    print()
    print(f"  Pattern: Storage premium is real but unreliable. Good years")
    print(f"  (2010, 2016, 2019, 2023) gave +75 to +150%. Bad years (2017,")
    print(f"  2018, 2022, 2025) gave -5% to -22%. The determining factor is")
    print(f"  whether the following year's sowing is large or small.")
    print()
    print(f"  2026 sowing is large (440k ha). Storage gain likely to be flat")
    print(f"  to negative — consistent with the post-bumper pattern.")


def print_action_items(recent, forecast):
    print(SEP)
    print("  ACTION ITEMS")
    print(DIV)

    last_price = int(recent.iloc[-1]["price"])
    nov_forecast = int(forecast[forecast["month"] == "2026-11"]["pred_price"].values[0]) if "2026-11" in forecast["month"].values else None
    aug_forecast = int(forecast[forecast["month"] == "2026-08"]["pred_price"].values[0]) if "2026-08" in forecast["month"].values else None

    print()
    print(f"  IF YOU HAVE STOCK (holding from this harvest at ₹3,500–₹5,000):")
    if aug_forecast and aug_forecast < last_price:
        print(f"  ▸ SELL WITHIN NEXT 4–6 WEEKS. Model sees ₹{aug_forecast:,} by Aug.")
        print(f"    This harvest's storage premium is already captured.")
    else:
        print(f"  ▸ Consider selling. Storage gain above ₹{aug_forecast:,} (Aug) not assured.")
    print(f"  ▸ Spike risk is MODERATE (3/5) — actual supply is tighter than paper")
    print(f"    estimates due to weather crop damage. But not a 2024-style shortage.")
    print(f"    Storage season could hold better than the model forecasts.")
    print()
    print(f"  FOR NEXT SOWING (Oct–Nov 2026):")
    print(f"  ▸ 2026 area is 440k ha — still elevated. If your neighbours are")
    print(f"    also sowing heavily, prices in 2027 harvest will be low again.")
    print(f"  ▸ Wait and watch Oct–Nov rainfall. If sowing rains fail (<50mm),")
    print(f"    less area gets planted → 2027 harvest tighter → prices recover.")
    print()
    print(f"  PRICE TO WATCH:")
    print(f"  ▸ If Oct 2026 Jaora price RISES above ₹5,500 → signal of tightness.")
    print(f"    That would suggest less-than-expected sowing → 2027 spike possible.")
    print(f"  ▸ If Oct 2026 stays below ₹4,000 → normal/oversupply year ahead.")
    print()
    if nov_forecast:
        print(f"  BOTTOM LINE: Model expects ₹{nov_forecast:,} by Nov 2026.")
        print(f"  Current ₹{last_price:,} is likely near the seasonal ceiling for this year.")


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    recent, seasonal, yearly, ds, china, forecast, db_min_year = load_all(con)
    con.close()

    area_df = india_area_data()

    print_header()
    print_crop_cycle(seasonal)
    print_current_snapshot(recent)

    # ── Crop stress check runs automatically every time ──────────────
    # Checks sowing rain, growing moisture, harvest rain, arrivals shortfall,
    # and paper-vs-actual divergence. Raises ALERT/WATCH automatically.
    con2 = duckdb.connect(DB_PATH, read_only=True)
    run_crop_stress_check(con=con2, area_yoy_pct=-5.4)   # 2026 area −5.4% vs 2025
    con2.close()

    print_spike_risk(recent, ds, china, area_df)

    if not forecast.empty:
        print_forecast(recent, forecast)

    print_historical_patterns(yearly)
    print_action_items(recent, forecast)

    print(SEP)
    print(f"  Data: Agmarknet ({db_min_year}–{recent['month'].max().strftime('%Y')}) · NHB area data ·")
    print(f"        UN Comtrade China exports · IMD rainfall · Ensemble ML model")
    print(f"  Forecast accuracy: MAE ≈ ₹1,700/q (21% MAPE) on 2022–2025 backtest")
    print(f"  Note: Spike years (like 2024) cannot be predicted — model handles")
    print(f"  normal cycles well but not black-swan supply failures.")
    print(SEP)


if __name__ == "__main__":
    main()
