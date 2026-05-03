#!/usr/bin/env python3
"""
External signals for garlic price prediction:
1. NEWS: Google News RSS + Krishak Jagat + ET (English + Hindi garlic queries)
2. CHINA: UN Comtrade garlic (070320) export volumes/prices as proxy for global supply
3. INDIA EXPORTS: UN Comtrade India garlic exports (APEDA equivalent)
4. Save to: garlic_news.csv, comtrade_garlic.csv, external_signals_summary.csv
"""
import requests, feedparser, duckdb, time, json, re
import pandas as pd
from datetime import datetime, date
from html.parser import HTMLParser

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
HEADERS = {"User-Agent": UA, "Accept": "*/*"}
DB_PATH = "garlic.duckdb"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────
# HTML stripping helper
# ─────────────────────────────────────────────────────────────────

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
    def get_text(self):
        return " ".join(self.parts).strip()

def strip_html(text):
    if not text:
        return ""
    p = HTMLStripper()
    try:
        p.feed(str(text))
        return p.get_text()[:500]
    except Exception:
        return str(text)[:500]


# ─────────────────────────────────────────────────────────────────
# 1. NEWS FEEDS
# ─────────────────────────────────────────────────────────────────

GARLIC_KEYWORDS = [
    "garlic", "lahsun", "lehsun", "lashun", "lahsoon",
    "safed", "white polish", "jaora", "mandsaur", "neemuch",
    "mandi bhav", "rabi", "kharif", "spice export",
]

NEWS_FEEDS = [
    # Google News RSS - English
    ("google_en_garlic",      "https://news.google.com/rss/search?q=garlic+India+price+mandi&hl=en-IN&gl=IN&ceid=IN:en"),
    ("google_en_garlic_exp",  "https://news.google.com/rss/search?q=garlic+India+export+price+2024+2025&hl=en-IN&gl=IN&ceid=IN:en"),
    ("google_en_garlic_mp",   "https://news.google.com/rss/search?q=garlic+Madhya+Pradesh+price+mandi&hl=en-IN&gl=IN&ceid=IN:en"),
    ("google_en_garlic_crop", "https://news.google.com/rss/search?q=garlic+crop+harvest+India+arrivals&hl=en-IN&gl=IN&ceid=IN:en"),
    ("google_en_china_garlic","https://news.google.com/rss/search?q=China+garlic+export+price+supply&hl=en&gl=US&ceid=US:en"),
    # Google News RSS - Hindi (garlic = लहसुन)
    ("google_hi_lahsun",      "https://news.google.com/rss/search?q=%E0%A4%B2%E0%A4%B9%E0%A4%B8%E0%A5%81%E0%A4%A8+%E0%A4%AD%E0%A4%BE%E0%A4%B5&hl=hi-IN&gl=IN&ceid=IN:hi"),
    ("google_hi_lahsun_mandi","https://news.google.com/rss/search?q=%E0%A4%B2%E0%A4%B9%E0%A4%B8%E0%A5%81%E0%A4%A8+%E0%A4%AE%E0%A4%82%E0%A4%A1%E0%A5%80&hl=hi-IN&gl=IN&ceid=IN:hi"),
    # Krishak Jagat - leading Hindi agri publication
    ("krishakjagat",          "https://krishakjagat.org/feed/"),
    # ET top stories (filter locally)
    ("et_topstories",         "https://economictimes.indiatimes.com/rssfeedstopstories.cms"),
    # ET Agriculture
    ("et_agri",               "https://economictimes.indiatimes.com/news/economy/agriculture/rssfeeds/14148469.cms"),
]

def is_garlic_relevant(title, summary=""):
    text = (title + " " + summary).lower()
    return any(kw in text for kw in GARLIC_KEYWORDS)

def fetch_all_news():
    log("=== PART 1: NEWS FEEDS ===")
    rows = []
    seen = set()
    for feed_name, url in NEWS_FEEDS:
        log(f"  Fetching {feed_name} ...")
        try:
            feed = feedparser.parse(url)
            total = len(feed.entries)
            relevant = 0
            for entry in feed.entries:
                title   = strip_html(entry.get("title", ""))
                summary = strip_html(entry.get("summary", entry.get("description", "")))
                link    = entry.get("link", "")
                # published date
                pub = entry.get("published", entry.get("updated", ""))
                try:
                    pub_dt = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d") if hasattr(entry, "published_parsed") and entry.published_parsed else ""
                except Exception:
                    pub_dt = ""
                # All Google News garlic feeds are already garlic-filtered; others need keyword check
                is_relevant = True if "google_en_garlic" in feed_name or "google_hi" in feed_name \
                              else is_garlic_relevant(title, summary)
                if not is_relevant:
                    continue
                key = title.strip().lower()[:80]
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "source":    feed_name,
                    "pub_date":  pub_dt,
                    "title":     title,
                    "summary":   summary[:300],
                    "url":       link,
                    "fetched_at": date.today().isoformat(),
                })
                relevant += 1
            log(f"    {feed_name}: {total} entries, {relevant} garlic-relevant")
        except Exception as e:
            log(f"    {feed_name}: ERROR {e}")
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pub_date", ascending=False)
        # Check if file exists to append or create
        try:
            existing = pd.read_csv("garlic_news.csv")
            combined = pd.concat([existing, df]).drop_duplicates(subset=["title"])
            combined.to_csv("garlic_news.csv", index=False)
            log(f"  Appended {len(df)} new articles → garlic_news.csv ({len(combined)} total)")
        except FileNotFoundError:
            df.to_csv("garlic_news.csv", index=False)
            log(f"  Saved {len(df)} articles → garlic_news.csv")
    else:
        log("  No news articles found")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. UN COMTRADE — China + India garlic trade
# ─────────────────────────────────────────────────────────────────
# UN Comtrade public/preview endpoint: free, no auth needed
# Rate limit: ~1 req/sec recommended; preview returns max 500 rows

COMTRADE_BASE = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

# HS codes for garlic:
#   070320 = Garlic, fresh or chilled
#   070310 = Onions & shallots (nearby)

HS_GARLIC = "070320"

def fetch_comtrade(reporter_code, reporter_name, flow="X", years=None, partner=0, delay=3.0):
    """Fetch Comtrade annual trade data. flow: X=exports, M=imports."""
    if years is None:
        years = list(range(2017, date.today().year + 1))

    period_str = ",".join(str(y) for y in years)
    url = COMTRADE_BASE
    params = {
        "reporterCode": reporter_code,
        "cmdCode": HS_GARLIC,
        "flowCode": flow,
        "period": period_str,
        "partnerCode": partner,   # 0 = World total
    }
    log(f"  Comtrade: {reporter_name} {flow} garlic, years {years[0]}–{years[-1]}, partner={partner}")
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            log(f"  Rate limited (429), waiting {delay*2}s...")
            time.sleep(delay * 2)
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        count = data.get("count", 0)
        rows  = data.get("data", [])
        log(f"    Got {count} rows")
        time.sleep(delay)
        return rows
    except Exception as e:
        log(f"    ERROR: {e}")
        return []

def build_comtrade_df(rows, reporter_name, flow_label):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Keep useful columns only
    keep = ["refYear", "reporterCode", "reporterDesc", "partnerCode", "partnerDesc",
            "flowCode", "cmdCode", "cmdDesc",
            "qty", "qtyUnitAbbr", "cifvalue", "fobvalue", "primaryValue"]
    df = df[[c for c in keep if c in df.columns]]
    df["reporter_name"] = reporter_name
    df["flow_label"]    = flow_label
    return df

def fetch_all_comtrade():
    log("=== PART 2: UN COMTRADE — GARLIC TRADE DATA ===")
    all_dfs = []

    # China exports to world (HS 070320)
    # China country code = 156
    china_exp = fetch_comtrade(156, "China", flow="X", partner=0, delay=3.0)
    all_dfs.append(build_comtrade_df(china_exp, "China", "exports_to_world"))

    # China exports to India specifically (partner=356)
    china_to_india = fetch_comtrade(156, "China", flow="X", partner=356, delay=3.0)
    all_dfs.append(build_comtrade_df(china_to_india, "China", "exports_to_India"))

    # India exports to world
    india_exp = fetch_comtrade(356, "India", flow="X", partner=0, delay=3.0)
    all_dfs.append(build_comtrade_df(india_exp, "India", "exports_to_world"))

    # India imports from China (to see how much Chinese garlic enters India)
    india_from_china = fetch_comtrade(356, "India", flow="M", partner=156, delay=3.0)
    all_dfs.append(build_comtrade_df(india_from_china, "India", "imports_from_China"))

    combined = pd.concat([df for df in all_dfs if not df.empty], ignore_index=True)

    if not combined.empty:
        combined.to_csv("comtrade_garlic.csv", index=False)
        log(f"  Saved {len(combined)} Comtrade rows → comtrade_garlic.csv")

        # Summary: one world-total row per year (max qty row per year)
        log("\n  === COMTRADE WORLD TOTALS (annual garlic exports) ===")
        for (reporter, flow_label), grp in combined.groupby(["reporter_name", "flow_label"]):
            log(f"\n  {reporter} — {flow_label}")
            annual = (grp.sort_values("qty", ascending=False)
                      .groupby("refYear", as_index=False).first()
                      .sort_values("refYear"))
            for _, row in annual.iterrows():
                qty = row.get("qty", 0)
                fob = row.get("fobvalue", row.get("primaryValue", 0))
                try:
                    qty_mt = float(qty) / 1000
                    price_per_kg = float(fob) / float(qty) if float(qty) > 0 else 0
                    log(f"    {int(row['refYear'])}: {qty_mt/1e6:.2f}M MT, FOB=${float(fob)/1e6:.0f}M, ~${price_per_kg:.3f}/kg")
                except Exception:
                    log(f"    {int(row['refYear'])}: qty={qty}, fob={fob}")
    else:
        log("  No Comtrade data returned (rate limited or not reported)")

    return combined


# ─────────────────────────────────────────────────────────────────
# 3. COMBINED SIGNALS ENRICHMENT
# ─────────────────────────────────────────────────────────────────

def build_combined_signals():
    log("=== PART 3: COMBINING ALL SIGNALS ===")
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        price_df = con.execute("""
            SELECT DATE_TRUNC('month', date)::DATE as month,
                   ROUND(AVG(modal_price), 0) as avg_price,
                   ROUND(SUM(arrivals), 0)    as total_arrivals
            FROM clean_garlic_prices
            WHERE market_id IN (1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727)
            GROUP BY 1 ORDER BY 1
        """).fetchdf()
        con.close()
        log(f"  Loaded {len(price_df)} monthly price rows")
    except Exception as e:
        log(f"  DuckDB error: {e}")
        return

    # Attach Comtrade annual data (China export qty/price by year)
    try:
        ct = pd.read_csv("comtrade_garlic.csv")
        china_world = ct[(ct["reporter_name"]=="China") & (ct["flow_label"]=="exports_to_world")].copy()
        china_world["year"] = china_world["refYear"].astype(int)
        china_world = china_world[["year","qty","fobvalue"]].rename(
            columns={"qty":"china_garlic_export_qty","fobvalue":"china_garlic_fob_usd"})
        price_df["year"] = pd.to_datetime(price_df["month"]).dt.year
        combined = price_df.merge(china_world, on="year", how="left")
        log(f"  Merged Comtrade data for {china_world['year'].nunique()} years")
    except FileNotFoundError:
        combined = price_df.copy()
        log("  No comtrade_garlic.csv, skipping China trade merge")

    # Attach weather (monthly)
    try:
        weather = pd.read_csv("weather_monthly.csv")
        weather["month"] = pd.to_datetime(weather["ym"].astype(str)).dt.to_period("M").dt.to_timestamp().dt.date
        combined["month"] = pd.to_datetime(combined["month"]).dt.date
        combined = combined.merge(
            weather[["month","total_rain_mm","avg_max_temp","avg_min_temp","avg_soil_moisture"]],
            on="month", how="left")
        log(f"  Merged weather data")
    except FileNotFoundError:
        log("  No weather_monthly.csv, skipping")

    combined.to_csv("external_signals_summary.csv", index=False)
    log(f"  Saved {len(combined)} rows → external_signals_summary.csv")

    # Print recent months for verification
    log("\n  === RECENT 6 MONTHS ===")
    recent = combined.sort_values("month", ascending=False).head(6)
    for _, r in recent.iterrows():
        log(f"  {r['month']}: price=₹{r.get('avg_price',0):,.0f}/q, arrivals={r.get('total_arrivals',0):,.0f}q, rain={r.get('total_rain_mm','?')}mm")


# ─────────────────────────────────────────────────────────────────
# 4. PRINT NEWS DIGEST
# ─────────────────────────────────────────────────────────────────

def print_news_digest(df):
    if df is None or df.empty:
        return
    log("\n=== PART 4: RECENT GARLIC NEWS DIGEST (last 30 articles) ===")
    recent = df.sort_values("pub_date", ascending=False).head(30)
    current_date = None
    for _, row in recent.iterrows():
        d = row.get("pub_date", "")
        if d != current_date:
            print(f"\n  [{d}]")
            current_date = d
        src = row.get("source","").replace("google_en_","EN/").replace("google_hi_","HI/")
        title = row.get("title","")[:100]
        print(f"    {src}: {title}")


# ─────────────────────────────────────────────────────────────────
# 5. CHINA GARLIC ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_china_impact():
    """Correlate China garlic export volumes/prices with India domestic prices."""
    try:
        ct = pd.read_csv("comtrade_garlic.csv")
        china_all = ct[(ct["reporter_name"]=="China") & (ct["flow_label"]=="exports_to_world")].copy()
        if china_all.empty:
            log("  No China export data available")
            return

        # Get annual world total: one row per year with the highest qty (= world aggregate)
        china_annual = (china_all.sort_values("qty", ascending=False)
                        .groupby("refYear", as_index=False)
                        .first()[["refYear","qty","fobvalue"]]
                        .rename(columns={"refYear":"year","qty":"china_qty_kg","fobvalue":"china_fob_usd"}))
        china_annual["china_qty_mt"]     = china_annual["china_qty_kg"] / 1000
        china_annual["china_fob_per_kg"] = china_annual["china_fob_usd"] / china_annual["china_qty_kg"]
        china_annual["year"] = china_annual["year"].astype(int)

        con = duckdb.connect(DB_PATH, read_only=True)
        india_annual = con.execute("""
            SELECT YEAR(date) as year,
                   ROUND(AVG(modal_price),0)      as india_avg_price,
                   ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (3,4,5)),0) as post_harvest_price,
                   ROUND(SUM(arrivals)/1e6, 2)    as total_arrivals_M
            FROM clean_garlic_prices
            WHERE market_id IN (1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727)
            GROUP BY 1 ORDER BY 1
        """).fetchdf()
        con.close()

        merged = india_annual.merge(china_annual, on="year", how="left")

        print("\n" + "─"*72)
        print("CHINA GARLIC EXPORTS vs INDIA DOMESTIC PRICE (Jaora cluster)")
        print("─"*72)
        print(f"{'Year':>4} | {'China qty (MT)':>14} | {'China $/kg':>10} | {'India ₹/q':>10} | {'India arrivals':>14}")
        print("─"*72)
        for _, r in merged.iterrows():
            try:
                qty_mt = r.get("china_qty_mt", float("nan"))
                fob_kg = r.get("china_fob_per_kg", float("nan"))
                qty_str = f"{qty_mt/1e6:.2f}M MT" if not pd.isna(qty_mt) else "  n/a"
                fob_str = f"${fob_kg:.3f}" if not pd.isna(fob_kg) else "   n/a"
                price_str = f"₹{int(r['india_avg_price']):,}" if not pd.isna(r['india_avg_price']) else "n/a"
                arr_str = f"{r['total_arrivals_M']:.2f}M" if not pd.isna(r['total_arrivals_M']) else "n/a"
                print(f"{int(r['year']):>4} | {qty_str:>14} | {fob_str:>10} | {price_str:>10} | {arr_str:>14}")
            except Exception as ex:
                print(f"{int(r.get('year','?')):>4} | {'n/a':>14} | {'n/a':>10} | {r.get('india_avg_price','?'):>10} | {'n/a':>14}")
        print("─"*72)

        # Correlation
        corr_data = merged[["india_avg_price","china_qty_mt","china_fob_per_kg"]].dropna()
        if len(corr_data) >= 4:
            corr_qty   = corr_data["india_avg_price"].corr(corr_data["china_qty_mt"])
            corr_price = corr_data["india_avg_price"].corr(corr_data["china_fob_per_kg"])
            print(f"\nCorrelation China export VOLUME ↔ India avg price: {corr_qty:.2f}")
            print(f"Correlation China FOB $/kg      ↔ India avg price: {corr_price:.2f}")
            print()
            print("Interpretation:")
            print("  China high FOB price → competitive parity → helps India exporters")
            print("  China high volume    → global supply pressure → may depress Indian export demand")

        # Save china_annual for future use
        china_annual.to_csv("china_garlic_comtrade.csv", index=False)
        log("  Saved china_garlic_comtrade.csv")

    except FileNotFoundError:
        log("  No comtrade_garlic.csv — skipping China analysis")
    except Exception as e:
        log(f"  China analysis error: {e}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

SEP = "\n" + "═"*72 + "\n"

def main():
    print(SEP)
    print("EXTERNAL SIGNALS — GARLIC PRICE INTELLIGENCE")
    print(f"Run date: {date.today()}")
    print(SEP)

    # 1. News
    news_df = fetch_all_news()

    # 2. Comtrade (China + India garlic trade)
    comtrade_df = fetch_all_comtrade()

    # 3. Combined signals CSV
    build_combined_signals()

    # 4. News digest
    print_news_digest(news_df)

    # 5. China impact analysis
    analyze_china_impact()

    print(SEP)
    log("DONE — outputs: garlic_news.csv, comtrade_garlic.csv, external_signals_summary.csv")


if __name__ == "__main__":
    main()
