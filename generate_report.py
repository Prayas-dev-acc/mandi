#!/usr/bin/env python3
"""
Generate report.html — self-contained market intelligence dashboard.
Run: python3 generate_report.py  →  opens report.html in browser.
"""
import duckdb, json, webbrowser, os
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from crop_stress import run_crop_stress_check

DB_PATH       = "garlic.duckdb"
JAORA_CLUSTER = [1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727]
CLUSTER_SQL   = ",".join(str(x) for x in JAORA_CLUSTER)
OUT_PATH      = Path("report.html")


# ── data loaders ─────────────────────────────────────────────────────

def india_area():
    data = [
        (2016,310,1580),(2017,321,1693),(2018,303,1611),(2019,358,2910),
        (2020,363,2925),(2021,386,3190),(2022,408,3208),(2023,431,3240),
        (2024,390,2800),(2025,465,3650),(2026,440,3400),
    ]
    return pd.DataFrame(data, columns=["crop_year","area_kha","prod_kmt"])

def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)

    hist = con.execute(f"""
        SELECT DATE_TRUNC('month', date)::DATE as month,
               ROUND(AVG(modal_price),0) as price,
               ROUND(SUM(arrivals),0)    as arrivals
        FROM clean_garlic_prices
        WHERE market_id IN ({CLUSTER_SQL})
        GROUP BY 1 ORDER BY 1 DESC LIMIT 24
    """).fetchdf()
    hist["month"] = pd.to_datetime(hist["month"])
    hist = hist.sort_values("month").reset_index(drop=True)

    seasonal = con.execute(f"""
        SELECT MONTH(date) as mo,
               ROUND(AVG(modal_price),0)  as avg_price,
               ROUND(AVG(arrivals),0)     as avg_arr
        FROM clean_garlic_prices
        WHERE market_id IN ({CLUSTER_SQL})
        GROUP BY 1 ORDER BY 1
    """).fetchdf()

    yearly = con.execute(f"""
        SELECT YEAR(date) as yr,
               ROUND(AVG(modal_price),0)                                    as yr_avg,
               ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (3,4)),0) as harv_price,
               ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (7,8,9)),0) as stor_price
        FROM clean_garlic_prices WHERE market_id IN ({CLUSTER_SQL})
        GROUP BY 1 ORDER BY 1
    """).fetchdf()

    db_start = con.execute("SELECT YEAR(MIN(date)) FROM garlic_prices").fetchone()[0]
    con.close()

    forecast = pd.read_csv("price_forecast.csv") if Path("price_forecast.csv").exists() else pd.DataFrame()

    ds = pd.read_csv("demand_supply_features.csv") if Path("demand_supply_features.csv").exists() else pd.DataFrame()
    if not ds.empty:
        ds["month"] = pd.to_datetime(ds["month"])

    china = pd.read_csv("china_garlic_comtrade.csv") if Path("china_garlic_comtrade.csv").exists() else pd.DataFrame()

    return hist, seasonal, yearly, forecast, ds, china, db_start


def load_weather_signals():
    try:
        w = pd.read_csv("weather_monthly.csv")
        w["month"] = pd.to_datetime(w["ym"].astype(str)).dt.to_period("M").dt.to_timestamp()
        w["yr"] = w["month"].dt.year; w["mo"] = w["month"].dt.month
        oct_row  = w[(w["yr"] == 2025) & (w["mo"] == 10)]
        sow_rain = float(oct_row["total_rain_mm"].values[0]) if len(oct_row) else 0.0
        oct_hist = w[w["mo"] == 10]["total_rain_mm"].mean()
        feb_row  = w[(w["yr"] == 2026) & (w["mo"] == 2)]
        feb_sm   = float(feb_row["avg_soil_moisture"].values[0]) if len(feb_row) else 0.0
        feb_hist = w[w["mo"] == 2]["avg_soil_moisture"].mean()
        return {"sow_rain": sow_rain, "sow_hist": oct_hist,
                "feb_sm": feb_sm, "feb_sm_hist": feb_hist}
    except Exception:
        return {}


def compute_spike_risk(hist, ds, china, area_df):
    area_10yr = area_df["area_kha"].mean()
    area_cur  = 440; area_prev = 465; prod_cur = 3400; prod_10yr = area_df["prod_kmt"].mean()
    area_diff = (area_cur - area_10yr) / area_10yr * 100
    prod_diff = (prod_cur - prod_10yr) / prod_10yr * 100
    area_yoy  = (area_cur - area_prev) / area_prev * 100

    last_price = int(hist.iloc[-1]["price"])
    peak_price = int(hist["price"].max())

    # Harvest season arrivals (Dec 2025–May 2026 vs Dec 2024–May 2025)
    harv_2026 = 99836; harv_2025 = 166834
    arr_yoy   = (harv_2026 - harv_2025) / harv_2025 * 100

    weather = load_weather_signals()

    score = 3.0
    # Paper supply
    if area_diff > 15:  score -= 1.0
    elif area_diff > 8: score -= 0.5
    if prod_diff > 20:  score -= 1.0
    elif prod_diff > 10: score -= 0.5
    # Arrivals shortfall (real supply signal)
    if arr_yoy < -35:   score += 1.5
    elif arr_yoy < -20: score += 1.0
    # Weather damage
    if weather:
        sow_ratio = weather["sow_rain"] / max(weather["sow_hist"], 1)
        if sow_ratio > 5:   score += 1.0
        elif sow_ratio > 2: score += 0.5
        sm_ratio = weather["feb_sm"] / max(weather["feb_sm_hist"], 0.01)
        if sm_ratio > 1.1:  score += 0.5
    # Post-spike
    if last_price / peak_price < 0.35:  score -= 1.0
    elif last_price / peak_price < 0.55: score -= 0.5
    if area_yoy < -10: score += 1.0
    elif area_yoy < -5: score += 0.5

    cur_spread = None; cur_kerala = None; china_fob = None
    if not ds.empty and "south_spread" in ds.columns:
        ss = ds["south_spread"].dropna()
        cur_spread = float(ss.iloc[-1])
        baseline   = ss.iloc[-12:-1].median() if len(ss) >= 12 else ss.median()
        if baseline > 0 and cur_spread / baseline > 1.2: score += 0.5
        cur_kerala = float(ds["kerala_price"].dropna().iloc[-1]) if "kerala_price" in ds.columns else None
    if not china.empty and "china_fob_per_kg" in china.columns:
        cv = china["china_fob_per_kg"].dropna()
        china_fob = float(cv.iloc[-1])
        if china_fob / float(cv.iloc[:-1].mean()) > 1.25: score += 0.5

    star = max(1, min(5, round(score)))
    risk_label = {1:"VERY LOW",2:"LOW",3:"MODERATE",4:"HIGH",5:"CRITICAL"}[star]

    # Build signal list — arrivals + weather first (most important)
    signals = [
        {"label": "Arrivals Dec '25–May '26 vs prior yr",
         "value": f"{harv_2026:,} MT",
         "flag": "alert" if arr_yoy < -30 else "watch",
         "note": f"{arr_yoy:+.0f}% vs last year ({harv_2025:,} MT) — real supply signal"},
    ]
    if weather:
        sr = weather["sow_rain"]; sh = weather["sow_hist"]
        signals.append({"label": "Oct 2025 sowing rain",
                        "value": f"{sr:.0f} mm",
                        "flag": "alert" if sr > 5*sh else "watch",
                        "note": f"Hist avg {sh:.0f}mm — waterlogging → uneven germination, yield loss"})
        sm = weather["feb_sm"]; smh = weather["feb_sm_hist"]
        signals.append({"label": "Feb 2026 soil moisture",
                        "value": f"{sm:.3f}",
                        "flag": "watch" if sm > smh * 1.1 else "neutral",
                        "note": f"Hist avg {smh:.3f} — wet bulb-fill → fungal disease risk"})
    signals += [
        {"label": "India sowing area 2026 (est)",
         "value": f"{int(area_cur):,}k ha",
         "flag": "low" if area_diff > 8 else "neutral",
         "note": f"{area_diff:+.0f}% vs 10yr avg — paper supply ample, doesn't reflect crop damage"},
        {"label": "Price vs recent peak",
         "value": f"₹{last_price:,} / ₹{peak_price:,}",
         "flag": "neutral",
         "note": f"Down {(1-last_price/peak_price)*100:.0f}% from peak — normalization phase"},
    ]
    if cur_spread:
        signals.append({"label":"South spread (Kerala–Jaora)",
                        "value":f"₹{int(cur_spread):,}/q",
                        "flag":"neutral",
                        "note":f"Kerala ₹{int(cur_kerala):,} — strong demand pull" if cur_kerala else "Demand pull from south"})
    if china_fob:
        signals.append({"label":"China FOB export price",
                        "value":f"${china_fob:.2f}/kg",
                        "flag":"watch" if china_fob > 1.2 else "neutral",
                        "note":"High China price → more export demand for Indian garlic"})

    return star, risk_label, signals


def build_chart_data(hist, forecast):
    hist_labels  = [r["month"].strftime("%b %y") for _, r in hist.iterrows()]
    hist_prices  = [int(r["price"]) for _, r in hist.iterrows()]
    hist_arrivals = [int(r["arrivals"]) for _, r in hist.iterrows()]

    fcast_labels = []
    fcast_mid    = []
    fcast_low    = []
    fcast_high   = []
    if not forecast.empty:
        for _, r in forecast.iterrows():
            fcast_labels.append(r["month"])
            fcast_mid.append(int(r["pred_price"]))
            fcast_low.append(int(r["price_low"]))
            fcast_high.append(int(r["price_high"]))

    return {
        "hist_labels":   hist_labels,
        "hist_prices":   hist_prices,
        "hist_arrivals": hist_arrivals,
        "fcast_labels":  fcast_labels,
        "fcast_mid":     fcast_mid,
        "fcast_low":     fcast_low,
        "fcast_high":    fcast_high,
    }


def build_seasonal_data(seasonal, today_month=None):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    s = seasonal.set_index("mo")
    prices = [int(s.loc[i+1,"avg_price"]) if (i+1) in s.index else 0 for i in range(12)]
    arrs   = [int(s.loc[i+1,"avg_arr"])   if (i+1) in s.index else 0 for i in range(12)]
    # correct phases by Jan-indexed month (0=Jan … 11=Dec)
    phases_jan = ["GROWING","GROWING","HARVEST","HARVEST","HARVEST","STORAGE",
                  "STORAGE","STORAGE","STORAGE","SOWING","SOWING","GROWING"]
    # reorder to Oct–Sep cycle
    order        = [9,10,11,0,1,2,3,4,5,6,7,8]
    cycle_months = [10,11,12,1,2,3,4,5,6,7,8,9]
    labels = [months[i]      for i in order]
    ph     = [phases_jan[i]  for i in order]
    pr     = [prices[i]      for i in order]
    ar     = [arrs[i]        for i in order]
    today_idx = cycle_months.index(today_month) if today_month and today_month in cycle_months else -1
    return {"labels": labels, "phases": ph, "prices": pr, "arrivals": ar, "today_month_idx": today_idx}


def analog_years(yearly):
    analogs = [2017,2018,2020,2022,2025]
    ctx = {
        2017:"After 2016 record premium (+150%) — glut followed",
        2018:"Continued oversupply — worst storage year on record",
        2020:"COVID disruption — storage still +49% (exception)",
        2022:"Bumper after stable 2021; storage flat",
        2025:"Crash from 2024 spike; record sowing → storage −14%",
    }
    rows = []
    for _, r in yearly[yearly["yr"].isin(analogs)].iterrows():
        h = int(r["harv_price"]) if pd.notna(r["harv_price"]) else None
        s = int(r["stor_price"]) if pd.notna(r["stor_price"]) else None
        gain = round((s-h)/h*100) if h and s else None
        rows.append({"yr":int(r["yr"]),"harv":h,"stor":s,"gain":gain,"ctx":ctx.get(int(r["yr"]),"")})
    return rows


# ── HTML template ─────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lahsun Price Intelligence — Jaora Mandi</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0b0e1a;--card:#131728;--card2:#1a1f35;
    --border:#232840;--text:#e2e8f0;--muted:#7c85a2;
    --orange:#f59e0b;--green:#10b981;--red:#ef4444;
    --blue:#3b82f6;--purple:#8b5cf6;--yellow:#fbbf24;
  }
  body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
       font-size:14px;line-height:1.6;min-height:100vh}

  /* header */
  .header{background:linear-gradient(135deg,#0f1535 0%,#1a2040 100%);
          border-bottom:1px solid var(--border);padding:20px 32px;
          display:flex;align-items:center;justify-content:space-between}
  .header-left h1{font-size:20px;font-weight:700;letter-spacing:.5px;
                  color:#fff}
  .header-left h1 span{color:var(--orange)}
  .header-left p{color:var(--muted);font-size:12px;margin-top:3px}
  .badge{background:var(--orange);color:#000;font-size:10px;font-weight:700;
         padding:3px 8px;border-radius:4px;letter-spacing:.5px}

  /* layout */
  .main{padding:24px 32px;max-width:1400px;margin:0 auto}
  .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px}
  .grid2{display:grid;grid-template-columns:3fr 2fr;gap:16px;margin-bottom:20px}
  .grid2r{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:20px}
  .full{margin-bottom:20px}
  @media(max-width:900px){.grid3,.grid2,.grid2r{grid-template-columns:1fr}}

  /* cards */
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}
  .card-title{font-size:10px;text-transform:uppercase;letter-spacing:1px;
              color:var(--muted);margin-bottom:12px;font-weight:600}
  .kpi-value{font-size:32px;font-weight:700;color:#fff;letter-spacing:-1px}
  .kpi-sub{font-size:12px;color:var(--muted);margin-top:4px}
  .kpi-change{display:inline-block;font-size:13px;font-weight:600;
              padding:2px 8px;border-radius:20px;margin-top:8px}
  .up{background:rgba(16,185,129,.15);color:var(--green)}
  .dn{background:rgba(239,68,68,.15);color:var(--red)}
  .nt{background:rgba(99,102,241,.15);color:#a5b4fc}

  /* spike risk */
  .stars{font-size:26px;letter-spacing:4px;margin:8px 0}
  .risk-label{font-size:13px;font-weight:600;margin-bottom:14px}
  .risk-low{color:var(--green)}
  .risk-mod{color:var(--yellow)}
  .risk-high{color:var(--red)}
  .signal-row{display:flex;align-items:flex-start;gap:10px;
              padding:8px 0;border-bottom:1px solid var(--border)}
  .signal-row:last-child{border-bottom:none}
  .sig-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:5px}
  .dot-low{background:var(--green)}
  .dot-neutral{background:var(--muted)}
  .dot-watch{background:var(--yellow)}
  .dot-alert{background:var(--red)}
  .sig-label{font-size:12px;color:var(--muted);min-width:170px;flex-shrink:0}
  .sig-value{font-size:12px;font-weight:600;color:#fff;min-width:110px}
  .sig-note{font-size:11px;color:var(--muted);flex:1}

  /* phase cards */
  .phase-card{border-radius:10px;padding:16px;border:1px solid var(--border)}
  .phase-card.active{border-color:rgba(255,255,255,.25);box-shadow:0 0 0 1px rgba(255,255,255,.1)}
  .phase-card-header{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;
                     margin-bottom:6px;display:flex;align-items:center;justify-content:space-between}
  .phase-sowing{background:rgba(251,191,36,.07)}
  .phase-sowing .phase-card-header{color:#fbbf24}
  .phase-growing{background:rgba(16,185,129,.07)}
  .phase-growing .phase-card-header{color:#10b981}
  .phase-harvest{background:rgba(239,68,68,.07)}
  .phase-harvest .phase-card-header{color:#ef4444}
  .phase-storage{background:rgba(99,102,241,.07)}
  .phase-storage .phase-card-header{color:#818cf8}
  .phase-months{font-size:13px;font-weight:600;color:#fff;margin-bottom:4px}
  .phase-price-range{font-size:11px;color:var(--muted);margin-bottom:8px}
  .phase-desc{font-size:12px;color:var(--muted);line-height:1.5;margin-bottom:10px}
  .phase-action{font-size:11px;font-weight:600;padding:3px 10px;border-radius:4px;display:inline-block}
  .phase-sowing  .phase-action{background:rgba(251,191,36,.15);color:#fbbf24}
  .phase-growing .phase-action{background:rgba(16,185,129,.15);color:#10b981}
  .phase-harvest .phase-action{background:rgba(239,68,68,.15);color:#ef4444}
  .phase-storage .phase-action{background:rgba(99,102,241,.15);color:#818cf8}
  .phase-here-badge{font-size:9px;font-weight:700;letter-spacing:.5px;
                    background:rgba(255,255,255,.15);color:#fff;padding:2px 8px;border-radius:10px}

  /* forecast table */
  table{width:100%;border-collapse:collapse}
  thead tr{background:var(--card2)}
  th{padding:10px 12px;text-align:left;font-size:11px;text-transform:uppercase;
     letter-spacing:.5px;color:var(--muted);font-weight:600}
  td{padding:10px 12px;border-bottom:1px solid var(--border);font-size:13px}
  tbody tr:last-child td{border-bottom:none}
  tbody tr:hover{background:rgba(255,255,255,.02)}
  .sell-sig{color:var(--red);font-weight:600}
  .flat-sig{color:var(--muted)}
  .buy-sig{color:var(--green);font-weight:600}
  .price-band{font-size:11px;color:var(--muted)}

  /* history table */
  .gain-pos{color:var(--green);font-weight:600}
  .gain-neg{color:var(--red);font-weight:600}

  /* action items */
  .action-block{margin-bottom:16px}
  .action-block h3{font-size:12px;text-transform:uppercase;letter-spacing:.5px;
                   color:var(--orange);font-weight:700;margin-bottom:10px}
  .action-item{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)}
  .action-item:last-child{border-bottom:none}
  .action-bullet{color:var(--orange);font-size:14px;flex-shrink:0;margin-top:1px}
  .action-text{font-size:13px;color:var(--text);line-height:1.5}

  /* chart container */
  .chart-wrap{position:relative;height:280px}

  /* footer */
  .footer{text-align:center;padding:20px 32px;color:var(--muted);font-size:11px;
          border-top:1px solid var(--border);margin-top:8px}
  .footer span{color:var(--orange)}

  /* section label */
  .section-label{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                 color:var(--orange);font-weight:700;margin-bottom:12px;
                 display:flex;align-items:center;gap:8px}
  .section-label::after{content:'';flex:1;height:1px;background:var(--border)}

  /* narrative card */
  .narrative{font-size:13px;color:var(--muted);line-height:1.7}
  .narrative b{color:var(--text)}
  .narrative .bullet{display:flex;gap:8px;margin:5px 0}
  .narrative .bullet span:first-child{color:var(--orange);flex-shrink:0}

  .tag{display:inline-block;padding:1px 7px;border-radius:4px;
       font-size:10px;font-weight:700;letter-spacing:.5px}
  .tag-storage{background:rgba(99,102,241,.2);color:#a5b4fc}
  .tag-harvest{background:rgba(239,68,68,.2);color:#fca5a5}
  .tag-sowing{background:rgba(251,191,36,.2);color:#fcd34d}
  .tag-growing{background:rgba(16,185,129,.2);color:#6ee7b7}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <h1>🧄 <span>LAHSUN</span> PRICE INTELLIGENCE</h1>
    <p>Jaora · Mandsaur · Neemuch · Piplya · Sailana cluster — Madhya Pradesh</p>
  </div>
  <div style="text-align:right">
    <div class="badge">v1 REPORT</div>
    <div style="color:var(--muted);font-size:11px;margin-top:6px">__DATE__</div>
  </div>
</div>

<div class="main">

  <!-- KPI CARDS -->
  <div class="grid3">
    <div class="card">
      <div class="card-title">Current Price</div>
      <div class="kpi-value">₹__PRICE__</div>
      <div class="kpi-sub">per quintal — Jaora cluster</div>
      <div class="kpi-change __MOM_CLASS__">__MOM__% vs last month</div>
      <div style="margin-top:8px;font-size:11px;color:var(--muted)">
        Year-on-year: <span style="color:var(--text)">__YOY__% vs May '25</span>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Spike Risk Indicator</div>
      <div class="stars">__STARS__</div>
      <div class="risk-label __RISK_CLASS__">__RISK_LABEL__</div>
      <div style="font-size:12px;color:var(--muted);line-height:1.5">
        Area data says supply ample, but actual mandi arrivals
        (Dec 2025–May 2026) are 40% below prior season.
      </div>
    </div>
    <div class="card">
      <div class="card-title">6-Month Outlook</div>
      <div style="margin-top:4px">
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border)">
          <span style="color:var(--muted);font-size:12px">Jun 2026</span>
          <span style="font-weight:600">₹__FC0__</span>
          <span class="kpi-change nt">→ FLAT</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border)">
          <span style="color:var(--muted);font-size:12px">Aug 2026</span>
          <span style="font-weight:600">₹__FC2__</span>
          <span class="kpi-change dn">↓ SELL</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0">
          <span style="color:var(--muted);font-size:12px">Nov 2026</span>
          <span style="font-weight:600">₹__FC5__</span>
          <span class="kpi-change dn">↓ SELL</span>
        </div>
      </div>
    </div>
  </div>

  <!-- PRICE CHART + FORECAST TABLE -->
  <div class="section-label">Price History &amp; Forecast</div>
  <div class="grid2">
    <div class="card">
      <div class="card-title">Jaora Cluster — 24-Month History + 6-Month Forecast (₹/quintal)</div>
      <div class="chart-wrap">
        <canvas id="priceChart"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Forecast Detail — 70% Confidence Band</div>
      <table>
        <thead>
          <tr><th>Month</th><th>Low</th><th>Median</th><th>High</th><th>Phase</th><th>Signal</th></tr>
        </thead>
        <tbody id="fcastBody"></tbody>
      </table>
    </div>
  </div>

  <!-- CROP CYCLE -->
  <div class="section-label">Garlic Crop Calendar</div>
  <div class="card full">
    <div class="card-title">MP Malwa — Annual cycle: price and arrivals move in opposite directions</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px" id="phaseCards"></div>
    <div style="padding:9px 14px;background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);
                border-radius:8px;font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.6">
      <b style="color:var(--orange)">Inverse law:</b>
      when mandi arrivals rise, price falls — always.
      Mar–Apr peak (~400 MT/month) → annual price floor.
      Oct–Nov trough → annual price peak.
      The price line (left axis) and arrival bars (right axis) move in opposite directions below.
    </div>
    <div class="chart-wrap" style="height:260px"><canvas id="cycleChart"></canvas></div>
    <div style="display:flex;gap:20px;margin-top:12px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted)">
        <div style="width:22px;height:2px;background:#f59e0b;border-radius:2px"></div>Avg Price ₹/quintal (left axis)
      </div>
      <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted)">
        <div style="width:12px;height:12px;background:rgba(129,140,248,.55);border-radius:2px"></div>Avg Arrivals MT/month — colored by phase (right axis)
      </div>
    </div>
  </div>

  <!-- SPIKE RISK DETAIL + NARRATIVE -->
  <div class="section-label">Market Context</div>
  <div class="grid2r">
    <div class="card">
      <div class="card-title">What Happened &amp; Why It Matters</div>
      <div class="narrative">
        <div class="bullet"><span>▸</span><span><b>2024 shortage:</b> India area fell to 390k ha (lowest in 5 yrs). Crop was weak → prices hit <b>₹23,176/q</b> in Nov 2024 — a 15-year record.</span></div>
        <div class="bullet"><span>▸</span><span><b>2025 response:</b> Farmers sowed a record <b>465k ha (+19%)</b>. Bumper harvest flooded mandis. Harvest price crashed to ₹3,590 (Apr 2025). Holding stock through storage season was a mistake — prices fell <b>−14%</b> from harvest to storage.</span></div>
        <div class="bullet"><span>▸</span><span><b>2026 sowing:</b> 440k ha — still 16% above 10-yr average. But <b>Oct 2025 had 119mm of rain</b> (vs 27mm historical average) — severe waterlogging during sowing caused uneven germination and yield losses.</span></div>
        <div class="bullet"><span>▸</span><span><b>Weather damage confirmed by arrivals:</b> Total mandi arrivals in the 2025-26 harvest season (Dec 2025–May 2026) were only <b>99,836 MT — 40% below the 2024-25 season</b> and lowest since COVID. This is data we already have. Paper supply says ample; actual supply that reached mandi is tighter.</span></div>
        <div class="bullet"><span>▸</span><span><b>Now (May 2026):</b> ₹5,260 near the 15-year median (₹5,100). South demand strong — Kerala ₹15,600. Storage season may hold better than model forecasts given tighter actual supply.</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Spike Risk Signals</div>
      <div id="signalRows"></div>
      <div style="margin-top:14px;padding:10px;background:rgba(251,191,36,.07);
                  border:1px solid rgba(251,191,36,.25);border-radius:8px;font-size:12px;
                  color:var(--muted);line-height:1.6">
        <b style="color:var(--yellow)">UPGRADE TO HIGH (4/5) if:</b><br>
        Jun–Sep arrivals stay below 15,000 MT/month, OR<br>
        Oct 2026 price rises above ₹5,500
      </div>
    </div>
  </div>

  <!-- CROP STRESS CHECK -->
  <div class="section-label">Automated Crop Stress Check</div>
  <div class="card full" id="stressCard">
    <div class="card-title">Phase-by-phase weather &amp; supply anomaly detection — runs on every report</div>
    <div id="stressBody" style="margin-top:4px"></div>
    <div style="margin-top:12px;font-size:11px;color:var(--muted)">
      Checks: sowing rain anomaly · growing-season moisture · harvest rain · arrivals YoY · paper-vs-actual divergence
    </div>
  </div>

  <!-- HISTORICAL ANALOGUES -->
  <div class="section-label">Historical Analogues</div>
  <div class="card full">
    <div class="card-title">Post-Bumper Years — Storage Season Performance (Jul–Sep avg vs Harvest avg)</div>
    <table>
      <thead>
        <tr><th>Year</th><th>Harvest price</th><th>Jul–Sep avg</th><th>Storage gain</th><th>Context</th></tr>
      </thead>
      <tbody id="analogBody"></tbody>
    </table>
    <div style="margin-top:16px;padding:12px;background:var(--card2);border-radius:8px;
                font-size:12px;color:var(--muted);line-height:1.7">
      <b style="color:var(--text)">Pattern:</b> In post-bumper years (2017, 2018, 2022, 2025), storage season gains were
      <span style="color:var(--red)">flat to negative (−5% to −23%)</span>.
      Good storage years (2010, 2016, 2019, 2023) followed <em>tight</em> crop years.
      The key driver is always next year's sowing area.
      <b style="color:var(--yellow)">2026 sowing is still large → storage gain for this year likely flat or negative.</b>
    </div>
  </div>

  <!-- ACTION ITEMS -->
  <div class="section-label">Action Items</div>
  <div class="grid3">
    <div class="card">
      <div class="action-block">
        <h3>If you have stock</h3>
        <div class="action-item">
          <span class="action-bullet">▸</span>
          <span class="action-text"><b>Sell within 4–6 weeks.</b> Model sees ₹4,250 by Aug. The storage premium for this harvest is already captured at the current ₹5,260.</span>
        </div>
        <div class="action-item">
          <span class="action-bullet">▸</span>
          <span class="action-text">Spike risk is <b style="color:var(--yellow)">MODERATE (3/5)</b> due to weather damage cutting actual harvest. Storage season may hold better than forecast. But another ₹23k spike needs a much larger shortage.</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="action-block">
        <h3>For next sowing (Oct–Nov)</h3>
        <div class="action-item">
          <span class="action-bullet">▸</span>
          <span class="action-text">2026 area is 440k ha — still elevated. If neighbours are also sowing heavily, 2027 harvest prices will be low again.</span>
        </div>
        <div class="action-item">
          <span class="action-bullet">▸</span>
          <span class="action-text">Watch Oct–Nov rainfall. If sowing rains fail (&lt;50mm), less area gets planted → 2027 harvest tighter → prices recover.</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="action-block">
        <h3>Price levels to watch</h3>
        <div class="action-item">
          <span class="action-bullet" style="color:var(--red)">▸</span>
          <span class="action-text"><b style="color:var(--red)">Oct above ₹5,500</b> → tightness signal. Less-than-expected sowing → 2027 spike possible. Reassess hold strategy.</span>
        </div>
        <div class="action-item">
          <span class="action-bullet" style="color:var(--green)">▸</span>
          <span class="action-text"><b style="color:var(--green)">Oct below ₹4,000</b> → normal/oversupply year. Focus on cost reduction, not price speculation.</span>
        </div>
      </div>
    </div>
  </div>

</div><!-- end main -->

<!-- FOOTER -->
<div class="footer">
  Data: <span>Agmarknet __DBSTART__–present</span> · NHB area &amp; production ·
  Open-Meteo weather · Ensemble ML (ElasticNet + XGBoost) · auto-retrained daily
  <br><span id="accuracyNote" style="display:inline-block;margin-top:4px">Accuracy: loading...</span>
</div>

<script>
const D = __CHART_DATA__;
const SEASONAL = __SEASONAL_DATA__;
const SIGNALS = __SIGNALS__;
const ANALOGS = __ANALOGS__;
const FCAST = __FCAST_TABLE__;
const STRESS = __STRESS_ALERTS__;
const ACCURACY = __ACCURACY__;

// ── Model Accuracy Badge (self-learning) ───────────────────────────
if (ACCURACY && ACCURACY.n_months_scored > 0) {
  const trendColor = ACCURACY.accuracy_trend === 'improving' ? '#10b981' : '#fbbf24';
  const mapeColor  = ACCURACY.mape_all < 20 ? '#10b981' : ACCURACY.mape_all < 30 ? '#fbbf24' : '#ef4444';
  const noteEl = document.getElementById('accuracyNote');
  if (noteEl) {
    noteEl.innerHTML = `
      <span style="color:${mapeColor};font-weight:700">MAPE ${ACCURACY.mape_all}%</span>
      &nbsp;·&nbsp; MAE ₹${Number(ACCURACY.mae_all).toLocaleString('en-IN')}
      &nbsp;·&nbsp; bias ₹${ACCURACY.bias_rs > 0 ? '+' : ''}${ACCURACY.bias_rs}
      &nbsp;·&nbsp; <span style="color:${trendColor}">${ACCURACY.accuracy_trend}</span>
      &nbsp;·&nbsp; ${ACCURACY.n_months_scored} months scored
      <br><span style="color:var(--muted)">${ACCURACY.calibration_note}</span>`;
  }
}

// ── Price Chart ────────────────────────────────────────────────────
const allLabels = [...D.hist_labels, ...D.fcast_labels];
const histLen   = D.hist_labels.length;

// Build arrays: null-pad history columns for forecast range and vice versa
const histPriceData = [...D.hist_prices, ...Array(D.fcast_labels.length).fill(null)];
const fcastMidData  = [...Array(histLen - 1).fill(null), D.hist_prices[histLen-1], ...D.fcast_mid];
const fcastLowData  = [...Array(histLen - 1).fill(null), D.hist_prices[histLen-1], ...D.fcast_low];
const fcastHighData = [...Array(histLen - 1).fill(null), D.hist_prices[histLen-1], ...D.fcast_high];

Chart.defaults.color = '#7c85a2';
Chart.defaults.borderColor = '#232840';

const ctx = document.getElementById('priceChart').getContext('2d');
const priceChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: allLabels,
    datasets: [
      {
        label: 'Actual price',
        data: histPriceData,
        borderColor: '#f59e0b',
        backgroundColor: 'transparent',
        borderWidth: 2.5,
        pointRadius: 3,
        pointBackgroundColor: '#f59e0b',
        tension: 0.3,
        spanGaps: false,
      },
      {
        label: 'Forecast (median)',
        data: fcastMidData,
        borderColor: '#3b82f6',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [6,3],
        pointRadius: 3,
        pointBackgroundColor: '#3b82f6',
        tension: 0.3,
        spanGaps: false,
      },
      {
        label: 'Forecast band (high)',
        data: fcastHighData,
        borderColor: 'rgba(99,102,241,.3)',
        backgroundColor: 'rgba(99,102,241,.08)',
        borderWidth: 1,
        borderDash: [3,3],
        pointRadius: 0,
        tension: 0.3,
        fill: '+1',
        spanGaps: false,
      },
      {
        label: 'Forecast band (low)',
        data: fcastLowData,
        borderColor: 'rgba(99,102,241,.3)',
        backgroundColor: 'rgba(99,102,241,.08)',
        borderWidth: 1,
        borderDash: [3,3],
        pointRadius: 0,
        tension: 0.3,
        fill: false,
        spanGaps: false,
      },
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {mode:'index',intersect:false},
    plugins: {
      legend: {
        labels: {boxWidth:12,font:{size:11},
                 filter: (item) => !item.text.includes('(high)') && !item.text.includes('(low)')}
      },
      tooltip: {
        backgroundColor: '#1a1f35',
        borderColor: '#232840',
        borderWidth: 1,
        callbacks: {
          label: (ctx) => ctx.parsed.y ? ` ₹${ctx.parsed.y.toLocaleString('en-IN')}` : null
        }
      }
    },
    scales: {
      x: {grid:{color:'#1e2340'},ticks:{font:{size:10},maxRotation:45}},
      y: {
        grid:{color:'#1e2340'},
        ticks:{
          font:{size:11},
          callback: v => '₹' + (v/1000).toFixed(0) + 'k'
        }
      }
    }
  }
});

// ── Forecast Table ─────────────────────────────────────────────────
const fcastBody = document.getElementById('fcastBody');
const BASE_PRICE = D.hist_prices[D.hist_prices.length - 1];
FCAST.forEach(r => {
  const vs = ((r.pred - BASE_PRICE) / BASE_PRICE * 100).toFixed(0);
  const sig = r.chg > 6 ? '<span class="buy-sig">↑ BUY</span>'
            : r.chg < -6 ? '<span class="sell-sig">↓ SELL</span>'
            : '<span class="flat-sig">→ FLAT</span>';
  const phClass = 'tag-' + r.phase.toLowerCase();
  fcastBody.innerHTML += `<tr>
    <td>${r.month}</td>
    <td class="price-band">₹${r.low.toLocaleString('en-IN')}</td>
    <td><b>₹${r.pred.toLocaleString('en-IN')}</b></td>
    <td class="price-band">₹${r.high.toLocaleString('en-IN')}</td>
    <td><span class="tag ${phClass}">${r.phase}</span></td>
    <td>${sig} <span style="color:var(--muted);font-size:11px">${vs}%</span></td>
  </tr>`;
});

// ── Crop Cycle ─────────────────────────────────────────────────────
const PHASE_DEFS = [
  {key:'SOWING',  label:'SOWING',  months:'Oct – Nov', cls:'phase-sowing',  idxs:[0,1],
   desc:'Seeds planted. Old stock squeeze → annual price peak. Limited fresh supply.', action:'Sell old stock'},
  {key:'GROWING', label:'GROWING', months:'Dec – Feb', cls:'phase-growing', idxs:[2,3,4],
   desc:'Bulbs forming underground. No new crop reaching mandi. Prices drift down slowly.', action:'Hold if storage cheap'},
  {key:'HARVEST', label:'HARVEST', months:'Mar – May', cls:'phase-harvest', idxs:[5,6,7],
   desc:'New crop floods mandis. Arrivals peak → price hits annual floor. Best time to buy for storage.', action:'Buy if next crop small'},
  {key:'STORAGE', label:'STORAGE', months:'Jun – Sep', cls:'phase-storage', idxs:[8,9,10,11],
   desc:'Fresh stock depletes month by month. Price recovers steadily toward Oct–Nov peak.', action:'Sell toward Oct–Nov'},
];
const todayIdx = (SEASONAL.today_month_idx !== undefined) ? SEASONAL.today_month_idx : -1;

// Phase narrative cards
const phaseCards = document.getElementById('phaseCards');
PHASE_DEFS.forEach(ph => {
  const phPrices = ph.idxs.map(i => SEASONAL.prices[i]);
  const prMin = Math.min(...phPrices).toLocaleString('en-IN');
  const prMax = Math.max(...phPrices).toLocaleString('en-IN');
  const isHere = ph.idxs.includes(todayIdx);
  const badge  = isHere ? `<span class="phase-here-badge">◄ NOW</span>` : '';
  phaseCards.innerHTML += `
    <div class="phase-card ${ph.cls}${isHere?' active':''}">
      <div class="phase-card-header">${ph.label} ${badge}</div>
      <div class="phase-months">${ph.months}</div>
      <div class="phase-price-range">₹${prMin} – ₹${prMax}/q avg</div>
      <div class="phase-desc">${ph.desc}</div>
      <span class="phase-action">${ph.action}</span>
    </div>`;
});

// Color maps by phase
const phColors = {SOWING:'rgba(251,191,36,.55)',GROWING:'rgba(16,185,129,.55)',
                  HARVEST:'rgba(239,68,68,.55)', STORAGE:'rgba(99,102,241,.55)'};
const phBorder = {SOWING:'#fbbf24',GROWING:'#10b981',HARVEST:'#ef4444',STORAGE:'#818cf8'};

// "You are here" vertical line plugin
const todayLinePlugin = {
  id:'todayLine',
  afterDraw(chart) {
    if (todayIdx < 0) return;
    const {ctx, chartArea:{top,bottom}, scales:{x}} = chart;
    if (!x) return;
    const xPos = x.getPixelForValue(todayIdx);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.3)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(xPos, top); ctx.lineTo(xPos, bottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,.65)';
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillText('◄ now', xPos + 4, top + 14);
    ctx.restore();
  }
};

new Chart(document.getElementById('cycleChart').getContext('2d'), {
  type: 'bar',
  plugins: [todayLinePlugin],
  data: {
    labels: SEASONAL.labels,
    datasets: [
      {type:'line', label:'Avg Price ₹/q', data:SEASONAL.prices,
       borderColor:'#f59e0b', backgroundColor:'transparent', borderWidth:2.5,
       pointRadius:5, pointBackgroundColor:SEASONAL.phases.map(p=>phBorder[p]),
       pointBorderColor:'transparent', tension:0.35, yAxisID:'yPrice', order:1},
      {type:'bar', label:'Avg Arrivals MT/month', data:SEASONAL.arrivals,
       backgroundColor:SEASONAL.phases.map(p=>phColors[p]),
       borderColor:SEASONAL.phases.map(p=>phBorder[p]),
       borderWidth:1, yAxisID:'yArr', order:2}
    ]
  },
  options:{
    responsive:true, maintainAspectRatio:false,
    interaction:{mode:'index',intersect:false},
    plugins:{
      legend:{labels:{boxWidth:12,font:{size:11}}},
      tooltip:{
        backgroundColor:'#1a1f35', borderColor:'#232840', borderWidth:1,
        callbacks:{
          title: items => `${SEASONAL.labels[items[0].dataIndex]} — ${SEASONAL.phases[items[0].dataIndex]}`,
          label: ctx => ctx.dataset.yAxisID==='yPrice'
            ? ` Price: ₹${ctx.parsed.y.toLocaleString('en-IN')}/q`
            : ` Arrivals: ${ctx.parsed.y} MT/month`
        }
      }
    },
    scales:{
      x:{grid:{color:'rgba(35,40,64,.5)'},ticks:{font:{size:11}}},
      yPrice:{position:'left', grid:{color:'#1e2340'},
              ticks:{callback:v=>'₹'+(v/1000).toFixed(0)+'k', font:{size:10}},
              title:{display:true, text:'Price ₹/quintal', color:'#f59e0b', font:{size:10}}},
      yArr:{position:'right', grid:{drawOnChartArea:false},
            ticks:{callback:v=>v+' MT', font:{size:10}},
            title:{display:true, text:'Arrivals MT/month', color:'#818cf8', font:{size:10}}}
    }
  }
});

// ── Spike Risk Signals ─────────────────────────────────────────────
const signalRows = document.getElementById('signalRows');
SIGNALS.forEach(s => {
  signalRows.innerHTML += `
    <div class="signal-row">
      <div class="sig-dot dot-${s.flag}"></div>
      <div>
        <div style="display:flex;gap:12px;align-items:baseline">
          <span class="sig-label">${s.label}</span>
          <span class="sig-value">${s.value}</span>
        </div>
        <div class="sig-note">${s.note}</div>
      </div>
    </div>`;
});

// ── Crop Stress Alerts ─────────────────────────────────────────────
const stressBody = document.getElementById('stressBody');
const phaseColors = {SOWING:'#f59e0b',GROWING:'#10b981',HARVEST:'#ef4444',SUPPLY:'#8b5cf6'};
const levelColors = {ALERT:'#ef4444',WATCH:'#fbbf24'};
if (STRESS.length === 0) {
  stressBody.innerHTML = `<div style="color:var(--green);font-size:13px;padding:8px 0">
    ✓ No stress signals detected. All crop phases within normal range.</div>`;
} else {
  const nAlert = STRESS.filter(a=>a.level==='ALERT').length;
  const nWatch = STRESS.filter(a=>a.level==='WATCH').length;
  stressBody.innerHTML += `<div style="margin-bottom:12px;font-size:13px">
    <span style="color:#ef4444;font-weight:700">${nAlert} ALERT${nAlert!==1?'s':''}</span>
    &nbsp;·&nbsp;
    <span style="color:#fbbf24;font-weight:600">${nWatch} WATCH signal${nWatch!==1?'s':''}</span>
  </div>`;
  let lastPhase = null;
  STRESS.forEach(a => {
    if (a.phase !== lastPhase) {
      lastPhase = a.phase;
      stressBody.innerHTML += `<div style="font-size:10px;text-transform:uppercase;
        letter-spacing:1px;color:${phaseColors[a.phase]||'#7c85a2'};
        font-weight:700;margin:12px 0 6px">${a.phase} PHASE</div>`;
    }
    const lc = levelColors[a.level] || '#7c85a2';
    stressBody.innerHTML += `
      <div style="display:flex;gap:10px;padding:7px 10px;margin-bottom:4px;
                  background:rgba(255,255,255,.02);border-radius:6px;
                  border-left:3px solid ${lc}">
        <span style="color:${lc};font-weight:700;font-size:11px;min-width:44px">${a.level}</span>
        <div>
          <div style="font-size:12px;font-weight:600;color:#fff">${a.metric}
            <span style="color:${lc};margin-left:8px">${a.actual}</span>
            <span style="color:var(--muted);font-weight:400"> vs ${a.baseline}</span>
          </div>
          <div style="font-size:11px;color:var(--muted);margin-top:2px">${a.message}</div>
        </div>
      </div>`;
  });
}

// ── Historical Analogues ───────────────────────────────────────────
const analogBody = document.getElementById('analogBody');
ANALOGS.forEach(r => {
  const gainClass = r.gain > 0 ? 'gain-pos' : 'gain-neg';
  const gainStr   = r.gain ? `${r.gain > 0 ? '+' : ''}${r.gain}%` : '—';
  analogBody.innerHTML += `<tr>
    <td><b>${r.yr}</b></td>
    <td>₹${(r.harv||0).toLocaleString('en-IN')}</td>
    <td>₹${(r.stor||0).toLocaleString('en-IN')}</td>
    <td class="${gainClass}">${gainStr}</td>
    <td style="color:var(--muted);font-size:12px">${r.ctx}</td>
  </tr>`;
});
</script>
</body>
</html>
"""

# ── main ─────────────────────────────────────────────────────────────

def main():
    hist, seasonal, yearly, forecast, ds, china, db_start = load_data()
    area_df  = india_area()

    # ── Run crop stress check (always, auto) ─────────────────────────
    con2 = duckdb.connect(DB_PATH, read_only=True)
    stress_alerts = run_crop_stress_check(con=con2, area_yoy_pct=-5.4, verbose=False)
    con2.close()

    star, risk_label, signals = compute_spike_risk(hist, ds, china, area_df)

    last  = hist.iloc[-1]
    prev  = hist.iloc[-2]
    yr_ago = hist[hist["month"].dt.month == last["month"].month]
    yr_ago = yr_ago.iloc[-2] if len(yr_ago) >= 2 else None

    mom  = (last["price"] - prev["price"]) / prev["price"] * 100
    yoy  = (last["price"] - yr_ago["price"]) / yr_ago["price"] * 100 if yr_ago is not None else 0
    mom_class = "up" if mom > 0 else "dn"

    stars_html = "★" * star + "☆" * (5 - star)
    risk_cls   = {"VERY LOW":"risk-low","LOW":"risk-low","MODERATE":"risk-mod",
                  "HIGH":"risk-high","CRITICAL":"risk-high"}.get(risk_label,"risk-mod")

    # forecast rows for JS
    fcast_rows = []
    prev_p = float(last["price"])
    for _, r in forecast.iterrows():
        p = int(r["pred_price"])
        chg = (p - prev_p) / prev_p * 100
        fcast_rows.append({"month": r["month"], "pred": p,
                           "low": int(r["price_low"]), "high": int(r["price_high"]),
                           "phase": r["season"], "chg": round(chg,1)})
        prev_p = p

    fc = [r["pred"] for r in fcast_rows]

    chart_data    = build_chart_data(hist, forecast)
    seasonal_data = build_seasonal_data(seasonal, today_month=date.today().month)
    analog_data   = analog_years(yearly)

    html = HTML.replace("__DATE__",       date.today().strftime("%d %b %Y"))
    html = html.replace("__PRICE__",      f"{int(last['price']):,}")
    html = html.replace("__MOM__",        f"{mom:+.1f}")
    html = html.replace("__MOM_CLASS__",  mom_class)
    html = html.replace("__YOY__",        f"{yoy:+.1f}")
    html = html.replace("__STARS__",      stars_html)
    html = html.replace("__RISK_LABEL__", risk_label)
    html = html.replace("__RISK_CLASS__", risk_cls)
    html = html.replace("__DBSTART__",    str(db_start))
    html = html.replace("__FC0__",  f"{fc[0]:,}" if len(fc)>0 else "—")
    html = html.replace("__FC2__",  f"{fc[2]:,}" if len(fc)>2 else "—")
    html = html.replace("__FC5__",  f"{fc[5]:,}" if len(fc)>5 else "—")

    html = html.replace("__CHART_DATA__",   json.dumps(chart_data))
    html = html.replace("__SEASONAL_DATA__",json.dumps(seasonal_data))
    html = html.replace("__SIGNALS__",      json.dumps(signals))
    html = html.replace("__ANALOGS__",      json.dumps(analog_data))
    html = html.replace("__FCAST_TABLE__",  json.dumps(fcast_rows))

    stress_js = [{"level": a.level, "phase": a.phase,
                  "metric": a.metric, "actual": a.actual,
                  "baseline": a.baseline, "message": a.message}
                 for a in stress_alerts]
    html = html.replace("__STRESS_ALERTS__", json.dumps(stress_js))

    # Accuracy summary (self-learning feedback loop)
    accuracy_path = Path("accuracy_summary.json")
    if accuracy_path.exists():
        acc = json.loads(accuracy_path.read_text())
    else:
        acc = {"status": "no_data", "n_months_scored": 0,
               "calibration_note": "Model running — accuracy tracking starts after first full month."}
    html = html.replace("__ACCURACY__", json.dumps(acc))

    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Generated: {OUT_PATH.resolve()}")
    import os
    if os.environ.get("CI") or "--no-browser" in __import__("sys").argv:
        return
    webbrowser.open(f"file://{OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
