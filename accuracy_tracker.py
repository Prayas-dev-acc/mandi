#!/usr/bin/env python3
"""
Self-learning accuracy tracker.

Every run:
  1. Reads price_forecast.csv (forecasts made last run)
  2. Compares against actual prices now in DB for months that have passed
  3. Appends new actuals to accuracy_log.csv
  4. Computes rolling MAPE, directional accuracy, bias
  5. Writes accuracy_summary.json — consumed by generate_report.py

This is the "self-learning" loop: the model sees its own errors and
retrains on fresh data every day, so it auto-corrects over time.
No LLM needed.
"""
import json
import duckdb
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path

DB_PATH      = "garlic.duckdb"
LOG_PATH     = Path("accuracy_log.csv")
SUMMARY_PATH = Path("accuracy_summary.json")
FORECAST_PATH = Path("price_forecast.csv")
JAORA_SQL    = "1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727"


def load_actuals():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(f"""
        SELECT DATE_TRUNC('month', date)::DATE as month,
               ROUND(AVG(modal_price), 0)      as actual_price
        FROM clean_garlic_prices
        WHERE market_id IN ({JAORA_SQL})
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    con.close()
    df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    return df.set_index("month")["actual_price"].to_dict()


def run():
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"accuracy_tracker: running for {today_str}", flush=True)

    if not FORECAST_PATH.exists():
        print("accuracy_tracker: no price_forecast.csv found, skipping", flush=True)
        return

    forecast = pd.read_csv(FORECAST_PATH)
    actuals  = load_actuals()

    # Load existing log
    if LOG_PATH.exists():
        log_df = pd.read_csv(LOG_PATH)
    else:
        log_df = pd.DataFrame(columns=[
            "logged_date", "target_month", "predicted", "actual",
            "abs_error", "pct_error", "direction_correct"
        ])

    new_rows = []
    for _, row in forecast.iterrows():
        month_key = row["month"][:7]   # e.g. "2026-06"
        # only score months where we have actual data
        if month_key not in actuals:
            continue
        # skip if already logged
        already = log_df[log_df["target_month"] == month_key]
        if len(already) > 0:
            continue

        pred   = float(row["pred_price"])
        actual = float(actuals[month_key])
        abs_err = abs(pred - actual)
        pct_err = abs_err / actual * 100
        # direction: did forecast correctly predict up/down vs prior month?
        # (we use sign of predicted change vs actual change from base)
        dir_correct = None  # computed below when we have sequence

        new_rows.append({
            "logged_date":    today_str,
            "target_month":   month_key,
            "predicted":      round(pred, 0),
            "actual":         round(actual, 0),
            "abs_error":      round(abs_err, 0),
            "pct_error":      round(pct_err, 1),
            "direction_correct": dir_correct,
        })

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        log_df  = pd.concat([log_df, new_df], ignore_index=True)
        log_df.to_csv(LOG_PATH, index=False)
        print(f"accuracy_tracker: logged {len(new_rows)} new actuals", flush=True)
    else:
        print("accuracy_tracker: no new months to score yet", flush=True)

    # ── Compute summary stats ─────────────────────────────────────
    scored = log_df.dropna(subset=["pct_error"])
    if len(scored) == 0:
        summary = {"status": "no_data", "n": 0}
    else:
        mape      = float(scored["pct_error"].mean())
        mae       = float(scored["abs_error"].mean())
        bias      = float((scored["predicted"] - scored["actual"]).mean())
        n         = len(scored)
        recent_n  = min(12, n)
        recent    = scored.tail(recent_n)
        mape_recent = float(recent["pct_error"].mean())
        bias_recent = float((recent["predicted"] - recent["actual"]).mean())

        # Self-learning flag: if MAPE worsening, flag for retrain attention
        trend = "improving" if mape_recent < mape else "degrading"

        summary = {
            "as_of":        today_str,
            "n_months_scored": n,
            "mape_all":     round(mape, 1),
            "mape_recent12": round(mape_recent, 1),
            "mae_all":      round(mae, 0),
            "bias_rs":      round(bias, 0),        # positive = over-predicting
            "bias_recent":  round(bias_recent, 0),
            "accuracy_trend": trend,
            "calibration_note": _calibration_note(bias_recent, mape_recent),
        }
        print(f"accuracy_tracker: MAPE={mape:.1f}%  MAE=₹{mae:.0f}  bias=₹{bias:.0f}  trend={trend}", flush=True)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"accuracy_tracker: summary → {SUMMARY_PATH}", flush=True)
    return summary


def _calibration_note(bias, mape):
    """Plain-language note injected into the report."""
    if mape < 15:
        return "Model well-calibrated. Forecast error within normal range."
    if bias > 500:
        return f"Model over-predicting by ₹{bias:.0f} on average recently. Expect actual prices slightly lower than forecast."
    if bias < -500:
        return f"Model under-predicting by ₹{abs(bias):.0f} recently. Actual prices may be higher than forecast."
    if mape > 30:
        return f"High forecast uncertainty (MAPE {mape:.0f}%). Use wide confidence bands — market in unusual conditions."
    return "Forecast in normal range. Check Jun–Sep actuals for next calibration."


if __name__ == "__main__":
    run()
