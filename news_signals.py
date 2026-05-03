#!/usr/bin/env python3
"""
Pull external signals that move garlic prices:
1. GDELT news — targeted queries for policy, export, China, drought events
2. Google Trends — "garlic price india" search interest (retail fear/demand signal)
3. Known government policy events (MEP, export ban, NAFED procurement) — documented
4. Cross-reference all signals with price data from DuckDB
"""

import requests, json, time, csv
import pandas as pd
import duckdb
from datetime import datetime, date
from pytrends.request import TrendReq
from concurrent.futures import ThreadPoolExecutor, as_completed

DB = "garlic.duckdb"
CLUSTER = [1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727]
CLUSTER_SQL = ",".join(str(x) for x in CLUSTER)

def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
SEP = "\n" + "═"*74
DIV = "\n" + "─"*74


# ─────────────────────────────────────────────────────────────────────────────
# 1. GDELT — targeted queries
# ─────────────────────────────────────────────────────────────────────────────

GDELT_QUERIES = [
    ("garlic export ban India",           "export_policy"),
    ("garlic minimum export price India", "export_policy"),
    ("China garlic production crop",      "china_supply"),
    ("garlic import China India",         "china_supply"),
    ("garlic price rise India",           "price_spike"),
    ("garlic shortage India",             "shortage"),
    ("Mandsaur garlic mandi",             "local"),
    ("Madhya Pradesh garlic price",       "local"),
    ("APEDA garlic export",               "export_data"),
]

def fetch_gdelt_year(query, category, year):
    """Fetch one year of GDELT results for a query."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query":         f"{query} sourcelang:english",
        "mode":          "artlist",
        "maxrecords":    "50",
        "format":        "json",
        "startdatetime": f"{year}0101000000",
        "enddatetime":   f"{year}1231235959",
        "sort":          "DateDesc",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return [{
                "date":     a.get("seendatetime", "")[:8],
                "title":    a.get("title", "")[:120],
                "url":      a.get("url", "")[:100],
                "domain":   a.get("domain", ""),
                "category": category,
                "query":    query,
            } for a in data.get("articles", [])]
    except Exception:
        pass
    return []


def fetch_gdelt_parallel(query, category):
    """Fetch all years in parallel threads."""
    articles = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_gdelt_year, query, category, yr): yr
                   for yr in range(2017, 2027)}
        for f in as_completed(futures):
            articles.extend(f.result() or [])
    return articles


log("Fetching GDELT news (parallel)...")
all_articles = []
for q, cat in GDELT_QUERIES:
    log(f"  Query: {q[:55]}...")
    arts = fetch_gdelt_parallel(q, cat)
    all_articles.extend(arts)
    log(f"    → {len(arts)} articles")
    time.sleep(0.5)

log(f"Total GDELT articles: {len(all_articles)}")

# Convert to DataFrame, deduplicate by URL
gdelt_df = pd.DataFrame(all_articles)
if len(gdelt_df):
    gdelt_df = gdelt_df.drop_duplicates(subset=["url"])
    gdelt_df["date_parsed"] = pd.to_datetime(gdelt_df["date"], format="%Y%m%d", errors="coerce")
    gdelt_df = gdelt_df.sort_values("date_parsed")
    gdelt_df.to_csv("news_gdelt.csv", index=False)
    log(f"Saved {len(gdelt_df)} unique articles to news_gdelt.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GOOGLE TRENDS — weekly search interest as demand/fear signal
# ─────────────────────────────────────────────────────────────────────────────

log("\nFetching Google Trends...")
pytrends = TrendReq(hl='en-IN', tz=330, timeout=(10,30))

TREND_KEYWORDS = [
    "garlic price india",   # retail consumer fear
    "lahsun price",         # Hindi — captures more regional searches
    "garlic export india",  # trader/exporter interest
]

trends_all = []
# Pull in 6-month chunks (pytrends limitation for weekly data)
periods = [
    ("2018-01-01", "2018-12-31"),
    ("2019-01-01", "2019-12-31"),
    ("2020-01-01", "2020-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", "2026-05-02"),
]

for kw in TREND_KEYWORDS:
    log(f"  Trends: '{kw}'")
    for start, end in periods:
        try:
            pytrends.build_payload([kw], cat=0,
                timeframe=f"{start} {end}", geo="IN")
            df = pytrends.interest_over_time()
            if len(df):
                df = df.reset_index()[["date", kw]]
                df.columns = ["date", "interest"]
                df["keyword"] = kw
                trends_all.append(df)
            time.sleep(1.5)
        except Exception as e:
            log(f"    ERROR {start}: {e}")
            time.sleep(5)

if trends_all:
    trends_df = pd.concat(trends_all, ignore_index=True)
    trends_df["date"] = pd.to_datetime(trends_df["date"]).dt.date
    # Pivot to wide
    trends_wide = trends_df.pivot_table(
        index="date", columns="keyword", values="interest", aggfunc="mean"
    ).reset_index()
    trends_wide.columns.name = None
    trends_wide.to_csv("google_trends_garlic.csv", index=False)
    log(f"Saved Google Trends to google_trends_garlic.csv ({len(trends_wide)} weeks)")
else:
    trends_wide = pd.DataFrame()
    log("No trends data fetched")


# ─────────────────────────────────────────────────────────────────────────────
# 3. KNOWN POLICY EVENTS — documented from public records
#    (MEP = Minimum Export Price, directly sets floor for domestic price)
# ─────────────────────────────────────────────────────────────────────────────

POLICY_EVENTS = [
    # Format: (date, event_type, description, price_impact)
    ("2017-06-01", "export",  "India garlic export season opens — MP harvest fully arrived", "neutral"),
    ("2018-01-01", "supply",  "National garlic glut — farmers over-planted post-2017 prices", "bearish"),
    ("2018-05-01", "supply",  "Desi garlic flood from MP/Gujarat — prices crash to ₹500/Q", "very_bearish"),
    ("2019-03-01", "weather", "Unseasonal rain in MP Mar-Apr 2019 — standing crop damaged in patches", "bullish"),
    ("2019-09-01", "export",  "China restricts garlic exports — India fills global demand gap", "very_bullish"),
    ("2019-10-01", "supply",  "Global garlic shortage — Indian exporters buying aggressively", "very_bullish"),
    ("2020-03-24", "policy",  "COVID-19 Nationwide Lockdown — mandis shut, arrivals collapse", "mixed"),
    ("2020-04-15", "policy",  "Partial mandi reopening — farmers dump stock in fear", "bearish"),
    ("2020-07-01", "export",  "Export demand recovers post-COVID — prices stabilise", "bullish"),
    ("2021-01-01", "supply",  "Good rabi crop 2020-21 — arrivals recover", "neutral"),
    ("2021-10-01", "export",  "Strong export demand from SE Asia and Europe", "bullish"),
    ("2022-01-01", "supply",  "Large MP crop — wet sowing 2021 led to bumper harvest", "bearish"),
    ("2022-06-01", "policy",  "Government imposes stock limits on garlic (Essential Commodities Act)", "bearish"),
    ("2022-09-01", "supply",  "Prices fall despite storage season — excess carry-over stock", "very_bearish"),
    ("2023-03-01", "weather", "Unseasonal rain Mar 2023 — Mandsaur/Neemuch crop partly damaged", "bullish"),
    ("2023-06-01", "export",  "China garlic crop shortfall — India becomes key supplier", "very_bullish"),
    ("2023-09-01", "export",  "APEDA reports record garlic export enquiries", "very_bullish"),
    ("2023-11-01", "supply",  "Cold storage stock running out — prices spike to ₹15,000+", "very_bullish"),
    ("2024-01-01", "supply",  "Very low carry-in stock + strong export demand — prices hit records", "very_bullish"),
    ("2024-03-01", "policy",  "Government monitors garlic prices amid retail inflation concerns", "bearish_risk"),
    ("2024-05-01", "export",  "MEP on garlic under discussion — export curb fear", "bearish_risk"),
    ("2024-08-01", "supply",  "Cold storage rapidly depleting — prices reach ₹30,000+/Q", "very_bullish"),
    ("2024-11-01", "supply",  "New kharif arrivals begin — price starts correcting from peak", "bearish"),
    ("2025-01-01", "supply",  "Fresh rabi 2024-25 crop arrives early — big supply surge expected", "bearish"),
    ("2025-02-01", "supply",  "Arrivals spike at Jaora/Mandsaur — harvest price crashes 66%", "very_bearish"),
    ("2025-06-01", "export",  "Export demand weak — China harvest normal, no India demand premium", "bearish"),
    ("2025-10-01", "supply",  "Sowing 2025 excellent (122mm rain) but area planted may be low post-crash", "neutral"),
    ("2026-01-01", "supply",  "Low 2026 arrivals confirm farmers reduced planting after 2025 crash", "bullish"),
    ("2026-04-01", "export",  "2026 export season opens — watch for China crop update", "watch"),
]

policy_df = pd.DataFrame(POLICY_EVENTS, columns=["date","event_type","description","price_impact"])
policy_df["date"] = pd.to_datetime(policy_df["date"]).dt.date
policy_df.to_csv("policy_events.csv", index=False)
log(f"\nSaved {len(policy_df)} policy events to policy_events.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORRELATE ALL SIGNALS WITH PRICE DATA
# ─────────────────────────────────────────────────────────────────────────────

log("\nLoading price data and correlating signals...")
con = duckdb.connect(DB, read_only=True)
monthly_price = con.execute(f"""
    SELECT DATE_TRUNC('month', date)::DATE as month,
           ROUND(AVG(modal_price),0) as avg_price,
           ROUND(SUM(arrivals),0) as total_arrivals,
           YEAR(date) as year, MONTH(date) as mo
    FROM clean_garlic_prices
    WHERE market_id IN ({CLUSTER_SQL})
    GROUP BY 1,4,5 ORDER BY 1
""").fetchdf()
con.close()
monthly_price["month"] = pd.to_datetime(monthly_price["month"]).dt.date


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRINT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print("EXTERNAL SIGNALS × GARLIC PRICE — JAORA CLUSTER")
print(SEP)

# A. Policy events aligned with price
print(DIV)
print("A. POLICY/SUPPLY EVENTS vs MONTHLY PRICE AT JAORA CLUSTER")
print(DIV)
print(f"{'Date':<12} {'Event':<12} {'Impact':<14} {'Price that month':>18}  Description")
print("─"*100)

for _, ev in policy_df.iterrows():
    ev_month = pd.Timestamp(ev["date"]).replace(day=1).date()
    matching = monthly_price[monthly_price["month"] == ev_month]
    price_str = f"₹{int(matching['avg_price'].iloc[0]):,}" if len(matching) else "no data"
    print(f"{str(ev['date']):<12} {ev['event_type']:<12} {ev['price_impact']:<14} {price_str:>18}  {ev['description'][:60]}")

# B. Google Trends peaks aligned with price
if len(trends_wide):
    print(DIV)
    print("B. GOOGLE TRENDS — 'garlic price india' search spikes (interest > 60)")
    print("   High search = consumers worried = price already rising or about to")
    print(DIV)
    col = "garlic price india" if "garlic price india" in trends_wide.columns else trends_wide.columns[1]
    spikes = trends_wide[trends_wide[col] > 60][["date", col]].copy()
    spikes["week_month"] = pd.to_datetime(spikes["date"]).dt.to_period("M").dt.to_timestamp().dt.date
    spikes_m = spikes.groupby("week_month")[col].max().reset_index()
    spikes_m.columns = ["month", "max_interest"]
    spikes_m = spikes_m.merge(
        monthly_price[["month","avg_price"]], on="month", how="left"
    )
    print(spikes_m.sort_values("month").to_string(index=False))

# C. GDELT top headlines by category
if len(gdelt_df):
    print(DIV)
    print("C. TOP GDELT NEWS EVENTS BY CATEGORY")
    print(DIV)
    for cat in ["export_policy","china_supply","shortage","price_spike","local","procurement"]:
        cat_df = gdelt_df[gdelt_df["category"] == cat].sort_values("date_parsed")
        if len(cat_df):
            print(f"\n  [{cat.upper()}] — {len(cat_df)} articles")
            for _, row in cat_df.head(8).iterrows():
                dt = str(row["date_parsed"])[:10] if pd.notna(row["date_parsed"]) else "unknown"
                print(f"    {dt}  {row['title'][:80]}")

# D. Trends correlation with price
if len(trends_wide) and "garlic price india" in trends_wide.columns:
    print(DIV)
    print("D. GOOGLE TRENDS vs PRICE CORRELATION (monthly)")
    print("   Does search interest LEAD or LAG actual price moves?")
    print(DIV)

    trends_wide["month"] = pd.to_datetime(trends_wide["date"]).dt.to_period("M").dt.to_timestamp().dt.date
    tm = trends_wide.groupby("month")["garlic price india"].mean().reset_index()
    tm.columns = ["month", "gtrend"]
    combined = monthly_price[["month","avg_price"]].merge(tm, on="month", how="inner")
    combined = combined.sort_values("month")

    # Check if trends lead price by 1 month
    combined["price_next_month"] = combined["avg_price"].shift(-1)
    combined["price_change_pct"] = combined["avg_price"].pct_change() * 100

    corr_same  = combined[["gtrend","avg_price"]].corr().iloc[0,1]
    corr_lead1 = combined[["gtrend","price_next_month"]].corr().iloc[0,1]

    print(f"  Correlation trends vs same-month price:   {corr_same:.2f}")
    print(f"  Correlation trends vs next-month price:   {corr_lead1:.2f}")
    print(f"  → {'Trends LEAD price (use as early signal)' if abs(corr_lead1) > abs(corr_same) else 'Trends COINCIDE with price (react together)'}")

    print(f"\n  Monthly view (interest vs price):")
    view = combined[["month","gtrend","avg_price","price_change_pct"]].dropna()
    view["gtrend"] = view["gtrend"].round(0)
    view["price_change_pct"] = view["price_change_pct"].round(1)
    print(view.tail(24).to_string(index=False))

# E. Composite signal summary
print(DIV)
print("E. COMPOSITE SIGNAL SUMMARY — ALL SIGNALS ALIGNED TO PRICE")
print(DIV)

wx_m = pd.read_csv("weather_monthly.csv")
wx_m["month"] = pd.to_datetime(wx_m.apply(
    lambda r: f"{int(r['year'])}-{int(r['month']):02d}-01", axis=1
)).dt.date

summary = monthly_price[["month","year","mo","avg_price","total_arrivals"]].copy()
summary = summary.merge(
    wx_m[["month","total_rain_mm","avg_max_temp"]], on="month", how="left"
)
if len(trends_wide):
    summary = summary.merge(tm, on="month", how="left")

summary["price_mom_chg"] = summary["avg_price"].pct_change() * 100

print(summary.sort_values("month").to_string(index=False))

# Save combined
summary.to_csv("combined_signals.csv", index=False)
log("\nSaved: combined_signals.csv, news_gdelt.csv, google_trends_garlic.csv, policy_events.csv")
