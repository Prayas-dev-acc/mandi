#!/usr/bin/env python3
"""
crop_stress.py  —  Automated crop stress detection for Jaora garlic.

Runs every time a report is generated. Compares real signals (arrivals,
weather) against historical baselines and raises alerts when they diverge
from paper supply estimates (area/production).

The core principle:
  Area sown  →  estimated supply   (paper, often wrong)
  Arrivals   →  actual supply      (real, always right)
  Gap between them  →  the alert zone

Three crop phases to monitor:
  SOWING  (Oct–Nov): heavy rain → waterlogging → lower effective area
  GROWING (Dec–Feb): excess moisture → fungal disease → yield loss
  HARVEST (Mar–May): unseasonal rain → field rot, quality damage

Import and call:  alerts = run_crop_stress_check(con)
"""
import duckdb
import numpy  as np
import pandas as pd
from pathlib import Path
from datetime import date

DB_PATH       = "garlic.duckdb"
CLUSTER_SQL   = "1085,522,182,2336,2088,2111,2662,2082,2727"

# ── thresholds ────────────────────────────────────────────────────────

# Rain anomaly: ratio of actual to historical monthly average
SOW_RAIN_WARN    = 2.5    # >2.5× historical Oct avg → watch
SOW_RAIN_ALERT   = 5.0    # >5×  historical Oct avg → alert (waterlogging)
HARV_RAIN_WARN   = 2.0    # >2×  historical Mar–Apr avg → watch (field damage)
HARV_RAIN_ALERT  = 4.0    # >4×  historical Mar–Apr avg → alert

# Soil moisture anomaly (ratio of actual to historical)
SM_WARN  = 1.10   # +10% above historical monthly avg → watch
SM_ALERT = 1.20   # +20% above → alert (disease risk)

# Arrivals shortfall vs same-season last year
ARR_YOY_WARN    = -0.20   # −20% → watch
ARR_YOY_ALERT   = -0.35   # −35% → alert

# Paper vs actual divergence: area up but arrivals down
DIVERGE_WARN    = 0.10    # area +10% but arrivals −20% → watch
DIVERGE_ALERT   = 0.15    # area +15% but arrivals −30% → alert


# ── data loaders ─────────────────────────────────────────────────────

def _load_weather(weather_csv="weather_monthly.csv"):
    try:
        w = pd.read_csv(weather_csv)
        w["month"] = pd.to_datetime(w["ym"].astype(str)).dt.to_period("M").dt.to_timestamp()
        w["yr"]    = w["month"].dt.year
        w["mo"]    = w["month"].dt.month
        return w
    except FileNotFoundError:
        return pd.DataFrame()


def _load_arrivals(con, years=5):
    """Harvest season arrivals by crop year.

    Window is Dec of prior calendar year through May of harvest year.
    E.g. crop_yr=2026 covers Dec 2025–May 2026 (all existing data).
    The Dec row is shifted: YEAR(date)+1 when MONTH=12 so it joins the right crop year.
    """
    df = con.execute(f"""
        SELECT
            YEAR(date) + CASE WHEN MONTH(date) = 12 THEN 1 ELSE 0 END AS crop_yr,
            MONTH(date) AS mo,
            ROUND(SUM(arrivals), 0) AS arr
        FROM clean_garlic_prices
        WHERE market_id IN ({CLUSTER_SQL})
          AND MONTH(date) IN (12, 1, 2, 3, 4, 5)
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).fetchdf()

    by_year = df.groupby("crop_yr")["arr"].sum().reset_index()
    by_year.columns = ["crop_yr", "total_arr"]
    return by_year


# ── single check ─────────────────────────────────────────────────────

class Alert:
    """One triggered stress alert."""
    def __init__(self, phase, metric, actual, baseline, ratio, level, message):
        self.phase    = phase    # SOWING / GROWING / HARVEST / SUPPLY
        self.metric   = metric
        self.actual   = actual
        self.baseline = baseline
        self.ratio    = ratio
        self.level    = level    # WATCH / ALERT
        self.message  = message

    def __repr__(self):
        sym = "▲▲" if self.level == "ALERT" else "▲ "
        return f"  [{self.level}] {sym} {self.phase:<8} {self.metric:<30} {self.message}"


def check_sowing_rain(w, crop_year):
    """Oct–Nov rainfall for the sowing season preceding crop_year harvest."""
    alerts = []
    if w.empty:
        return alerts

    sow_yr = crop_year - 1   # sowing happens Oct-Nov of previous calendar year
    for mo, mo_name in [(10, "Oct"), (11, "Nov")]:
        row  = w[(w["yr"] == sow_yr) & (w["mo"] == mo)]
        hist = w[w["mo"] == mo]["total_rain_mm"]
        if row.empty or hist.empty:
            continue
        actual   = float(row["total_rain_mm"].values[0])
        baseline = float(hist.mean())
        if baseline < 0.5:
            baseline = 0.5   # avoid division by near-zero
        ratio    = actual / baseline

        if ratio >= SOW_RAIN_ALERT:
            alerts.append(Alert(
                "SOWING", f"{mo_name} rain",
                f"{actual:.0f}mm", f"{baseline:.0f}mm avg", ratio, "ALERT",
                f"{actual:.0f}mm vs {baseline:.0f}mm avg ({ratio:.1f}×) — "
                f"severe waterlogging risk, uneven germination likely"
            ))
        elif ratio >= SOW_RAIN_WARN:
            alerts.append(Alert(
                "SOWING", f"{mo_name} rain",
                f"{actual:.0f}mm", f"{baseline:.0f}mm avg", ratio, "WATCH",
                f"{actual:.0f}mm vs {baseline:.0f}mm avg ({ratio:.1f}×) — "
                f"above-normal sowing rain, monitor germination"
            ))
    return alerts


def check_growing_moisture(w, crop_year):
    """Dec–Feb soil moisture (bulb-fill phase)."""
    alerts = []
    if w.empty:
        return alerts

    for yr_offset, mo, mo_name in [(-1, 12, "Dec"), (0, 1, "Jan"), (0, 2, "Feb")]:
        check_yr = crop_year + yr_offset
        row  = w[(w["yr"] == check_yr) & (w["mo"] == mo)]
        hist = w[w["mo"] == mo]["avg_soil_moisture"]
        if row.empty or hist.empty:
            continue
        actual   = float(row["avg_soil_moisture"].values[0])
        baseline = float(hist.mean())
        ratio    = actual / baseline if baseline > 0 else 1.0

        if ratio >= SM_ALERT:
            alerts.append(Alert(
                "GROWING", f"{mo_name} soil moisture",
                f"{actual:.3f}", f"{baseline:.3f} avg", ratio, "ALERT",
                f"{actual:.3f} vs {baseline:.3f} avg ({ratio:.2f}×) — "
                f"high moisture during bulb-fill, significant fungal disease risk"
            ))
        elif ratio >= SM_WARN:
            alerts.append(Alert(
                "GROWING", f"{mo_name} soil moisture",
                f"{actual:.3f}", f"{baseline:.3f} avg", ratio, "WATCH",
                f"{actual:.3f} vs {baseline:.3f} avg ({ratio:.2f}×) — "
                f"above-normal moisture, watch for purple blotch / white rot"
            ))
    return alerts


def check_harvest_rain(w, crop_year):
    """Mar–Apr rainfall during standing crop and harvest."""
    alerts = []
    if w.empty:
        return alerts

    for mo, mo_name in [(3, "Mar"), (4, "Apr")]:
        row  = w[(w["yr"] == crop_year) & (w["mo"] == mo)]
        hist = w[w["mo"] == mo]["total_rain_mm"]
        if row.empty or hist.empty:
            continue
        actual   = float(row["total_rain_mm"].values[0])
        baseline = float(hist.mean())
        if baseline < 0.5:
            baseline = 0.5
        ratio    = actual / baseline

        if ratio >= HARV_RAIN_ALERT:
            alerts.append(Alert(
                "HARVEST", f"{mo_name} rain",
                f"{actual:.1f}mm", f"{baseline:.1f}mm avg", ratio, "ALERT",
                f"{actual:.1f}mm vs {baseline:.1f}mm avg ({ratio:.1f}×) — "
                f"heavy rain during harvest, field rot and quality damage likely"
            ))
        elif ratio >= HARV_RAIN_WARN:
            alerts.append(Alert(
                "HARVEST", f"{mo_name} rain",
                f"{actual:.1f}mm", f"{baseline:.1f}mm avg", ratio, "WATCH",
                f"{actual:.1f}mm vs {baseline:.1f}mm avg ({ratio:.1f}×) — "
                f"above-normal harvest rain, quality risk"
            ))
    return alerts


def check_arrivals_shortfall(by_year, crop_year):
    """Harvest arrivals vs prior year and 3-year average."""
    alerts = []
    cur  = by_year.loc[by_year["crop_yr"] == crop_year, "total_arr"]
    prev = by_year.loc[by_year["crop_yr"] == crop_year - 1, "total_arr"]
    if cur.empty or prev.empty:
        return alerts

    cur_val  = float(cur.values[0])
    prev_val = float(prev.values[0])
    yoy      = (cur_val - prev_val) / prev_val

    # 3-year average
    hist_3 = by_year[by_year["crop_yr"].isin(
        [crop_year - 1, crop_year - 2, crop_year - 3])]["total_arr"]
    avg3 = float(hist_3.mean()) if len(hist_3) >= 2 else prev_val
    vs_avg3 = (cur_val - avg3) / avg3

    if yoy <= ARR_YOY_ALERT:
        alerts.append(Alert(
            "SUPPLY", "Harvest arrivals YoY",
            f"{cur_val:,.0f} MT", f"{prev_val:,.0f} MT last yr", yoy, "ALERT",
            f"{yoy:.0%} vs last year — real supply significantly below estimates"
        ))
    elif yoy <= ARR_YOY_WARN:
        alerts.append(Alert(
            "SUPPLY", "Harvest arrivals YoY",
            f"{cur_val:,.0f} MT", f"{prev_val:,.0f} MT last yr", yoy, "WATCH",
            f"{yoy:.0%} vs last year — monitor storage season for tightness"
        ))

    if vs_avg3 <= -0.30:
        alerts.append(Alert(
            "SUPPLY", "Arrivals vs 3yr avg",
            f"{cur_val:,.0f} MT", f"{avg3:,.0f} MT 3yr avg", vs_avg3, "ALERT",
            f"{vs_avg3:.0%} below 3yr avg — structural supply shortfall signal"
        ))
    return alerts


def check_paper_actual_divergence(area_yoy_pct, arr_yoy_pct):
    """Flag when area is up but arrivals are down — the key gap signal."""
    alerts = []
    # area_yoy_pct: % change in sown area (positive = more sown)
    # arr_yoy_pct:  % change in harvest arrivals (negative = less arrived)
    divergence = area_yoy_pct / 100 - arr_yoy_pct / 100   # e.g. +0.16 - (-0.40) = 0.56

    if area_yoy_pct > 0 and arr_yoy_pct < ARR_YOY_ALERT * 100:
        if divergence >= DIVERGE_ALERT:
            alerts.append(Alert(
                "SUPPLY", "Paper vs actual divergence",
                f"area {area_yoy_pct:+.0f}%, arrivals {arr_yoy_pct:+.0f}%",
                "area ≈ arrivals", divergence, "ALERT",
                f"Area sown UP {area_yoy_pct:+.0f}% but arrivals DOWN {arr_yoy_pct:.0f}% — "
                f"crop damage reduced realized yield significantly"
            ))
        elif divergence >= DIVERGE_WARN:
            alerts.append(Alert(
                "SUPPLY", "Paper vs actual divergence",
                f"area {area_yoy_pct:+.0f}%, arrivals {arr_yoy_pct:+.0f}%",
                "area ≈ arrivals", divergence, "WATCH",
                f"Area up but arrivals lagging — check weather and on-ground reports"
            ))
    return alerts


# ── main entry ────────────────────────────────────────────────────────

def run_crop_stress_check(con=None, crop_year=None, weather_csv="weather_monthly.csv",
                          area_yoy_pct=None, verbose=True):
    """
    Run all stress checks for a given crop year (default: current).
    Returns list of Alert objects sorted by severity.

    Parameters
    ----------
    con           : duckdb connection (opened if None)
    crop_year     : harvest year to check (default: current year)
    area_yoy_pct  : % change in sown area vs prior year (for divergence check)
    verbose       : print results to terminal
    """
    if crop_year is None:
        today = date.today()
        # Crop year = calendar year of Mar-Apr harvest
        crop_year = today.year if today.month >= 3 else today.year - 1

    close_con = False
    if con is None:
        con = duckdb.connect(DB_PATH, read_only=True)
        close_con = True

    w        = _load_weather(weather_csv)
    by_year  = _load_arrivals(con)
    if close_con:
        con.close()

    all_alerts = []
    all_alerts += check_sowing_rain(w, crop_year)
    all_alerts += check_growing_moisture(w, crop_year)
    all_alerts += check_harvest_rain(w, crop_year)
    all_alerts += check_arrivals_shortfall(by_year, crop_year)

    if area_yoy_pct is not None:
        cur_arr  = by_year.loc[by_year["crop_yr"] == crop_year, "total_arr"]
        prev_arr = by_year.loc[by_year["crop_yr"] == crop_year - 1, "total_arr"]
        if not cur_arr.empty and not prev_arr.empty:
            arr_yoy = (float(cur_arr.values[0]) - float(prev_arr.values[0])) \
                      / float(prev_arr.values[0]) * 100
            all_alerts += check_paper_actual_divergence(area_yoy_pct, arr_yoy)

    # Sort: ALERT before WATCH, then by phase order
    phase_order = {"SOWING": 0, "GROWING": 1, "HARVEST": 2, "SUPPLY": 3}
    all_alerts.sort(key=lambda a: (0 if a.level == "ALERT" else 1,
                                   phase_order.get(a.phase, 9)))

    if verbose:
        _print_stress_report(all_alerts, crop_year, by_year, w)

    return all_alerts


def _print_stress_report(alerts, crop_year, by_year, w):
    SEP = "\n" + "═" * 68
    DIV = "─" * 68
    print(SEP)
    print(f"  CROP STRESS CHECK — {crop_year} harvest")
    print(DIV)

    # Quick data freshness check
    if not w.empty:
        last_weather = w["month"].max()
        months_stale = (pd.Timestamp.now() - last_weather).days // 30
        if months_stale > 2:
            print(f"\n  ⚠  WEATHER DATA IS {months_stale} MONTHS OLD — run fetch_weather.py to update")
        else:
            print(f"\n  Weather data current to: {last_weather.strftime('%b %Y')}")

    last_arr = by_year[by_year["crop_yr"] <= crop_year].tail(3)
    if not last_arr.empty:
        print(f"  Arrivals (Dec of prior year – May of harvest year):")
        for _, r in last_arr.iterrows():
            yr   = int(r["crop_yr"])
            window = f"Dec {yr-1}–May {yr}"
            marker = " ◄ current" if yr == crop_year else ""
            print(f"    {yr} ({window}): {int(r['total_arr']):>10,} MT{marker}")

    if not alerts:
        print(f"\n  ✓ No stress signals detected for {crop_year} crop. All metrics within normal range.")
        return

    n_alerts = sum(1 for a in alerts if a.level == "ALERT")
    n_watch  = sum(1 for a in alerts if a.level == "WATCH")
    print(f"\n  {n_alerts} ALERT(s)  ·  {n_watch} WATCH signal(s)\n")

    current_phase = None
    for a in alerts:
        if a.phase != current_phase:
            current_phase = a.phase
            print(f"  ── {a.phase} PHASE {'─'*(50-len(a.phase))}")
        print(repr(a))
    print()

    # Impact summary
    supply_alerts = [a for a in alerts if a.phase == "SUPPLY" and a.level == "ALERT"]
    weather_alerts = [a for a in alerts if a.phase != "SUPPLY" and a.level == "ALERT"]

    if supply_alerts or weather_alerts:
        print(f"  IMPACT ASSESSMENT:")
        if weather_alerts:
            phases = list({a.phase for a in weather_alerts})
            print(f"  ▸ Weather stress detected in {', '.join(phases)} phase(s).")
            print(f"    This can reduce realized yield even when sown area is large.")
            print(f"    Area/production estimates should be treated with caution.")
        if supply_alerts:
            arr_a = next((a for a in supply_alerts if "YoY" in a.metric), None)
            if arr_a:
                print(f"  ▸ Arrivals signal confirms supply shortfall:")
                print(f"    {arr_a.message}")
        print(f"  ▸ Spike risk elevated. Monitor storage season arrivals closely.")
        print(f"    If Jun–Sep arrivals stay below 15,000 MT/month → HIGH risk.")


# ── standalone run ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else None
    # 2026 area was down ~5% from 2025 (440k vs 465k)
    alerts = run_crop_stress_check(crop_year=yr, area_yoy_pct=-5.4)
    print(f"\n  Total alerts raised: {len(alerts)}")
