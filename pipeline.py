#!/usr/bin/env python3
"""
Daily autonomous pipeline — runs at 6 AM IST via GitHub Actions.

Steps (each step failure is logged but does NOT abort the pipeline):
  1. Pull last 14 days of Agmarknet data into DuckDB
  2. Update weather_monthly.csv from Open-Meteo
  3. Retrain price prediction model → price_forecast.csv
  4. Track forecast accuracy → accuracy_log.csv + accuracy_summary.json
  5. Regenerate HTML report (headless)

Run locally:  python pipeline.py
Run in CI:    python pipeline.py --ci
"""
import sys, time, traceback
from datetime import datetime
from pathlib import Path

CI_MODE = "--ci" in sys.argv


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def step(name, fn):
    log(f"▶ {name}")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        log(f"  ✓ {name} ({elapsed:.0f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  ✗ {name} FAILED ({elapsed:.0f}s): {e}")
        traceback.print_exc()
        return None


def main():
    log("=" * 60)
    log(f"LAHSUN PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    log(f"Mode: {'CI/headless' if CI_MODE else 'local'}")
    log("=" * 60)

    # ── 1. Pull new Agmarknet data ────────────────────────────────
    def pull_data():
        import daily_fetch
        return daily_fetch.run(lookback_days=14)
    step("Agmarknet data pull (last 14 days)", pull_data)

    # ── 2. Update weather ─────────────────────────────────────────
    def update_wx():
        import update_weather
        return update_weather.run()
    step("Weather update (Open-Meteo)", update_wx)

    # ── 3. Retrain model ──────────────────────────────────────────
    def retrain():
        import subprocess
        r = subprocess.run(
            [sys.executable, "predict_price.py"],
            capture_output=True, text=True, timeout=600
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])
        # Print last 5 lines of model output
        for line in r.stdout.strip().split("\n")[-5:]:
            log(f"    {line}")
        return True
    step("Retrain prediction model", retrain)

    # ── 4. Track accuracy (self-learning) ─────────────────────────
    def track_accuracy():
        import accuracy_tracker
        return accuracy_tracker.run()
    step("Accuracy tracking / self-calibration", track_accuracy)

    # ── 5. Generate report ────────────────────────────────────────
    def gen_report():
        # Patch webbrowser to be a no-op in CI
        if CI_MODE:
            import webbrowser
            webbrowser.open = lambda *a, **kw: None
        import importlib, generate_report
        importlib.reload(generate_report)   # pick up patched webbrowser
        generate_report.main()
        return True
    step("Generate HTML report", gen_report)

    # ── Done ──────────────────────────────────────────────────────
    log("=" * 60)
    log("Pipeline complete.")

    # Print accuracy summary if available
    summary_path = Path("accuracy_summary.json")
    if summary_path.exists():
        import json
        s = json.loads(summary_path.read_text())
        if s.get("n_months_scored", 0) > 0:
            log(f"Model accuracy: MAPE {s['mape_all']}%  MAE ₹{s['mae_all']:,}  bias ₹{s['bias_rs']:+,}  trend={s['accuracy_trend']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
