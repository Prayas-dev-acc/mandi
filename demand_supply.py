#!/usr/bin/env python3
"""
Demand-Supply spread analysis for Jaora garlic.
Supply: Jaora + Mandsaur + Neemuch (competing MP source mandis)
Demand: Kerala, Tamil Nadu, Maharashtra, Telangana destination markets

Key insight: destination_price - source_price = trader margin.
Wide margin → traders rush to buy from MP → Jaora price rises 2-4 weeks later.
"""
import warnings; warnings.filterwarnings("ignore")
import duckdb, numpy as np, pandas as pd
from datetime import date, datetime

DB_PATH = "garlic.duckdb"
DIV = "─" * 72
SEP = "\n" + "═" * 72 + "\n"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Market definitions ────────────────────────────────────────────
SOURCE_MARKETS = {
    "Jaora":     [1085],
    "Mandsaur":  [522],
    "Neemuch":   [182],
    "Other_MP":  [2336, 2088, 2111, 2662, 2082, 2727],
}
ALL_SOURCE_IDS = [id for ids in SOURCE_MARKETS.values() for id in ids]

DEST_MARKETS = {
    "Kerala":      [135, 2687, 2683, 2684, 1081],
    "Tamil Nadu":  [1469, 2422, 4097, 4149, 4099, 4064],
    "Maharashtra": [164, 160, 2704, 1637, 3450],
    "Telangana":   [3760, 2],
}
ALL_DEST_IDS = [id for ids in DEST_MARKETS.values() for id in ids]


# ─────────────────────────────────────────────────────────────────
# 1. LOAD MONTHLY PRICES
# ─────────────────────────────────────────────────────────────────

def load_all_prices():
    con = duckdb.connect(DB_PATH, read_only=True)

    # Source: each mandi separately
    src_rows = []
    for name, ids in SOURCE_MARKETS.items():
        df = con.execute(f"""
            SELECT DATE_TRUNC('month', date)::DATE as month,
                   ROUND(AVG(modal_price), 0) as price,
                   ROUND(SUM(arrivals), 0)    as arrivals
            FROM clean_garlic_prices
            WHERE market_id IN ({','.join(str(x) for x in ids)})
            GROUP BY 1 ORDER BY 1
        """).fetchdf()
        df["source"] = name
        src_rows.append(df)
    src_df = pd.concat(src_rows)
    src_df["month"] = pd.to_datetime(src_df["month"])

    # Destination: each state separately
    dest_rows = []
    for state, ids in DEST_MARKETS.items():
        df = con.execute(f"""
            SELECT DATE_TRUNC('month', date)::DATE as month,
                   ROUND(AVG(modal_price), 0) as price,
                   ROUND(SUM(arrivals), 0)    as arrivals
            FROM clean_garlic_prices
            WHERE market_id IN ({','.join(str(x) for x in ids)})
            GROUP BY 1 ORDER BY 1
        """).fetchdf()
        df["state"] = state
        dest_rows.append(df)
    dest_df = pd.concat(dest_rows)
    dest_df["month"] = pd.to_datetime(dest_df["month"])

    con.close()
    return src_df, dest_df


# ─────────────────────────────────────────────────────────────────
# 2. BUILD MONTHLY SPREAD TABLE
# ─────────────────────────────────────────────────────────────────

def build_spread_table(src_df, dest_df):
    # MP combined source price (Jaora+Mandsaur+Neemuch average)
    mp_monthly = (src_df.groupby("month")
                  .agg(mp_price=("price","mean"), mp_arrivals=("arrivals","sum"))
                  .reset_index())

    # Jaora alone
    jaora_monthly = (src_df[src_df["source"]=="Jaora"]
                     .rename(columns={"price":"jaora_price","arrivals":"jaora_arrivals"})
                     [["month","jaora_price","jaora_arrivals"]])

    # Destination pivot
    dest_pivot = dest_df.pivot_table(
        index="month", columns="state", values="price", aggfunc="mean"
    ).reset_index()
    dest_pivot.columns.name = None
    dest_pivot = dest_pivot.rename(columns={
        "Kerala":      "kerala_price",
        "Tamil Nadu":  "tn_price",
        "Maharashtra": "mh_price",
        "Telangana":   "tg_price",
    })
    # Avg of Kerala + TN (premium buyers of Safed variety)
    kcols = [c for c in ["kerala_price","tn_price"] if c in dest_pivot.columns]
    if kcols:
        dest_pivot["south_premium_price"] = dest_pivot[kcols].mean(axis=1)

    # Merge everything
    df = mp_monthly.merge(jaora_monthly, on="month", how="left")
    df = df.merge(dest_pivot, on="month", how="left")

    # Spreads (destination - MP source)
    for col, label in [("kerala_price","kerala_spread"),
                        ("tn_price","tn_spread"),
                        ("mh_price","mh_spread"),
                        ("south_premium_price","south_spread")]:
        if col in df.columns:
            df[label] = df[col] - df["mp_price"]

    # Spreads lagged (can spread predict future Jaora price?)
    df["south_spread_lag1"] = df["south_spread"].shift(1)
    df["south_spread_lag2"] = df["south_spread"].shift(2)
    df["mp_arrivals_lag1"]  = df["mp_arrivals"].shift(1)

    df = df.sort_values("month").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 3. PRINT SPREAD ANALYSIS
# ─────────────────────────────────────────────────────────────────

def print_spread_analysis(df):
    print(SEP)
    print("DEMAND-SUPPLY SPREAD ANALYSIS  (₹/quintal)")
    print("MP Source (Jaora+Mandsaur+Neemuch) vs Destination Markets")
    print(SEP)

    # Recent 18 months
    recent = df.tail(18).copy()
    print(f"\n  {'Month':<10} {'MP src':>8} {'Jaora':>8} {'Kerala':>8} {'TN':>8} "
          f"{'Maha':>8} {'South↑':>8}  Margin signal")
    print(f"  {DIV}")
    for _, r in recent.iterrows():
        mp   = int(r.get("mp_price", 0) or 0)
        jar  = int(r.get("jaora_price", 0) or 0)
        ker  = int(r.get("kerala_price", 0) or 0)
        tn   = int(r.get("tn_price", 0) or 0)
        mh   = int(r.get("mh_price", 0) or 0)
        spr  = r.get("south_spread", np.nan)
        spr_str = f"₹{int(spr):+,}" if not pd.isna(spr) else "  n/a"
        sig  = ("🔥 STRONG BUY" if not pd.isna(spr) and spr > 8000 else
                "↑ BUY"        if not pd.isna(spr) and spr > 4000 else
                "→ HOLD"       if not pd.isna(spr) and spr > 1500 else
                "↓ WEAK"       if not pd.isna(spr) else "  —")
        print(f"  {str(r['month'].date())[:7]:<10} {mp:>8,} {jar:>8,} "
              f"{ker:>8,} {tn:>8,} {mh:>8,} {spr_str:>8}  {sig}")


def print_competitor_comparison(src_df):
    print(SEP)
    print("JAORA vs MANDSAUR vs NEEMUCH  (who is cheapest = who gets the buyers)")
    print(SEP)

    pivot = src_df[src_df["source"].isin(["Jaora","Mandsaur","Neemuch"])].pivot_table(
        index="month", columns="source", values=["price","arrivals"], aggfunc="first"
    )
    pivot.columns = [f"{b}_{a.lower()}" for a,b in pivot.columns]
    pivot = pivot.reset_index().sort_values("month")

    recent = pivot.tail(18)
    print(f"\n  {'Month':<10} {'Jaora ₹':>9} {'Msaur ₹':>9} {'Neemch ₹':>9} "
          f"{'Cheapest':>12}  {'Jaora arr':>10} {'Msaur arr':>10}")
    print(f"  {DIV}")
    for _, r in recent.iterrows():
        j_p  = r.get("Jaora_price", np.nan)
        m_p  = r.get("Mandsaur_price", np.nan)
        n_p  = r.get("Neemuch_price", np.nan)
        j_a  = r.get("Jaora_arrivals", 0)
        m_a  = r.get("Mandsaur_arrivals", 0)

        prices = {k: v for k, v in [("Jaora",j_p),("Mandsaur",m_p),("Neemuch",n_p)]
                  if not pd.isna(v)}
        cheapest = min(prices, key=prices.get) if prices else "—"
        cheapest_str = f"★ {cheapest}" if cheapest == "Jaora" else cheapest

        j_str = f"₹{int(j_p):,}" if not pd.isna(j_p) else "  n/a"
        m_str = f"₹{int(m_p):,}" if not pd.isna(m_p) else "  n/a"
        n_str = f"₹{int(n_p):,}" if not pd.isna(n_p) else "  n/a"
        print(f"  {str(r['month'].date())[:7]:<10} {j_str:>9} {m_str:>9} {n_str:>9} "
              f"{cheapest_str:>12}  {int(j_a or 0):>10,} {int(m_a or 0):>10,}")


# ─────────────────────────────────────────────────────────────────
# 4. SPREAD → PRICE CORRELATION (does spread predict Jaora price?)
# ─────────────────────────────────────────────────────────────────

def spread_predictive_power(df):
    print(SEP)
    print("DOES DESTINATION SPREAD PREDICT JAORA PRICE? (correlation analysis)")
    print(DIV)

    analysis = df[["month","jaora_price","south_spread",
                   "south_spread_lag1","south_spread_lag2",
                   "mp_arrivals","mp_arrivals_lag1"]].dropna()

    corr_same   = analysis["jaora_price"].corr(analysis["south_spread"])
    corr_lag1   = analysis["jaora_price"].corr(analysis["south_spread_lag1"])
    corr_lag2   = analysis["jaora_price"].corr(analysis["south_spread_lag2"])
    corr_arr    = analysis["jaora_price"].corr(analysis["mp_arrivals"])
    corr_arr_l1 = analysis["jaora_price"].corr(analysis["mp_arrivals_lag1"])

    print(f"\n  South spread (same month)    → Jaora price:  {corr_same:+.2f}")
    print(f"  South spread (1 month lag)   → Jaora price:  {corr_lag1:+.2f}  ← lead signal")
    print(f"  South spread (2 month lag)   → Jaora price:  {corr_lag2:+.2f}")
    print(f"  MP total arrivals            → Jaora price:  {corr_arr:+.2f}")
    print(f"  MP arrivals (1 month lag)    → Jaora price:  {corr_arr_l1:+.2f}")

    print(f"\n  Interpretation:")
    if abs(corr_lag1) > 0.5:
        print(f"  ✓ Spread from LAST month is a strong predictor of Jaora price TODAY")
        print(f"  ✓ When Kerala/TN pay much more than MP → traders pull stock from MP")
        print(f"    → supply tightens → Jaora price rises ~4 weeks later")
    elif abs(corr_lag1) > 0.3:
        print(f"  ~ Spread has moderate predictive power (lag-1 corr = {corr_lag1:.2f})")
    else:
        print(f"  ~ Spread correlation is weak — prices move more on local supply")

    # Year-by-year spread vs Jaora price
    print(f"\n  ANNUAL AVERAGE: Spread vs Jaora price")
    print(f"  {'Year':>5} | {'South spread':>13} | {'Jaora price':>12} | {'MP arrivals':>12}")
    print(f"  {'-'*55}")
    annual = analysis.copy()
    annual["year"] = pd.to_datetime(annual["month"]).dt.year
    for yr, grp in annual.groupby("year"):
        spr = grp["south_spread"].mean()
        jp  = grp["jaora_price"].mean()
        arr = grp["mp_arrivals"].mean()
        print(f"  {int(yr):>5} | ₹{int(spr):>11,} | ₹{int(jp):>10,} | {int(arr):>12,} q/mo")

    return corr_lag1


# ─────────────────────────────────────────────────────────────────
# 5. CURRENT SITUATION SNAPSHOT
# ─────────────────────────────────────────────────────────────────

def current_snapshot(df, src_df):
    print(SEP)
    print("CURRENT SITUATION (May 2026) — WHAT IT MEANS FOR YOUR DECISION")
    print(SEP)

    last  = df.dropna(subset=["jaora_price"]).iloc[-1]
    prev  = df.dropna(subset=["jaora_price"]).iloc[-2]

    j_price   = last.get("jaora_price", 0)
    mp_price  = last.get("mp_price", 0)
    ker_price = last.get("kerala_price", np.nan)
    tn_price  = last.get("tn_price", np.nan)
    mh_price  = last.get("mh_price", np.nan)
    spread    = last.get("south_spread", np.nan)
    prev_spr  = prev.get("south_spread", np.nan)

    print(f"\n  SOURCE PRICES (this month, MP):")
    print(f"    Jaora price   :  ₹{int(j_price or 0):,}/q")
    print(f"    MP avg (all 3):  ₹{int(mp_price or 0):,}/q")

    print(f"\n  DESTINATION PRICES (what buyers are paying):")
    if not pd.isna(ker_price): print(f"    Kerala        :  ₹{int(ker_price):,}/q")
    if not pd.isna(tn_price):  print(f"    Tamil Nadu    :  ₹{int(tn_price):,}/q")
    if not pd.isna(mh_price):  print(f"    Maharashtra   :  ₹{int(mh_price):,}/q")

    print(f"\n  TRADER MARGIN (after ~₹700-900/q transport cost):")
    if not pd.isna(spread):
        net_margin = spread - 800
        print(f"    South spread  :  ₹{int(spread):,}/q")
        print(f"    Net margin est:  ₹{int(net_margin):,}/q  (after transport)")
        if net_margin > 5000:
            print(f"    → VERY HIGH margin — expect strong buying pressure from traders")
        elif net_margin > 2000:
            print(f"    → HEALTHY margin — steady demand from south traders likely")
        elif net_margin > 500:
            print(f"    → THIN margin — some buying but traders cautious")
        else:
            print(f"    → No margin — south buyers may wait or look elsewhere")

    if not pd.isna(prev_spr) and not pd.isna(spread):
        spr_chg = spread - prev_spr
        print(f"\n  Spread vs last month: {spr_chg:+,.0f}  "
              f"({'widening → more buying pressure' if spr_chg > 0 else 'narrowing → less pressure'})")

    # MP total arrivals trend
    mp_arr = last.get("mp_arrivals", 0)
    prev_arr = prev.get("mp_arrivals", 0)
    print(f"\n  TOTAL MP ARRIVALS (Jaora+Mandsaur+Neemuch combined):")
    print(f"    This month:  {int(mp_arr or 0):,} q")
    print(f"    Last month:  {int(prev_arr or 0):,} q")
    if prev_arr and mp_arr:
        chg = ((mp_arr - prev_arr) / prev_arr) * 100
        print(f"    Change:      {chg:+.0f}%  "
              f"({'↓ tightening supply' if chg < -20 else '↑ supply building' if chg > 20 else '→ stable'})")

    # Who is cheapest right now
    latest_src = src_df[src_df["source"].isin(["Jaora","Mandsaur","Neemuch"])]
    latest_src = latest_src[latest_src["month"] == latest_src["month"].max()]
    print(f"\n  COMPETITION (who is cheapest right now):")
    for _, r in latest_src.sort_values("price").iterrows():
        star = " ← cheapest, gets buyers first" if r["price"] == latest_src["price"].min() else ""
        print(f"    {r['source']:<12}: ₹{int(r['price'] or 0):,}/q  ({int(r['arrivals'] or 0):,} q arrivals){star}")

    print(f"\n  BOTTOM LINE:")
    if not pd.isna(spread) and spread > 8000:
        print(f"  🔥 Spread is ₹{int(spread):,}/q — extremely high. South traders have strong")
        print(f"     incentive to buy from MP. Expect buying pressure to push Jaora prices")
        print(f"     up over the next 4-8 weeks. HOLD or sell gradually at higher prices.")
    elif not pd.isna(spread) and spread > 4000:
        print(f"  ↑  Spread is ₹{int(spread):,}/q — healthy. Steady demand from south.")
        print(f"     Prices likely to hold or rise slightly. No rush to sell everything now.")
    elif not pd.isna(spread) and spread > 1500:
        print(f"  →  Spread is ₹{int(spread):,}/q — moderate. Demand is there but thin.")
        print(f"     Sell at market price, don't expect a big rally.")
    else:
        print(f"  ↓  Spread is thin. South buyers not incentivised to buy from MP.")
        print(f"     Consider selling soon rather than holding.")


# ─────────────────────────────────────────────────────────────────
# 6. SAVE ENRICHED FEATURES FOR MODEL
# ─────────────────────────────────────────────────────────────────

def save_features(df):
    cols = ["month","mp_price","jaora_price","jaora_arrivals","mp_arrivals",
            "kerala_price","tn_price","mh_price","tg_price",
            "kerala_spread","tn_spread","mh_spread","south_spread",
            "south_spread_lag1","south_spread_lag2","mp_arrivals_lag1"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.to_csv("demand_supply_features.csv", index=False)
    log(f"Saved {len(out)} rows → demand_supply_features.csv")
    log("These features are ready to be added to the prediction model.")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    log("Loading prices from DuckDB...")
    src_df, dest_df = load_all_prices()
    log(f"  Source: {len(src_df)} rows | Destination: {len(dest_df)} rows")

    log("Building spread table...")
    spread_df = build_spread_table(src_df, dest_df)

    print_spread_analysis(spread_df)
    print_competitor_comparison(src_df)
    corr = spread_predictive_power(spread_df)
    current_snapshot(spread_df, src_df)
    save_features(spread_df)

    print(SEP)
    log("DONE — run predict_price.py next to use these features in the model")


if __name__ == "__main__":
    main()
