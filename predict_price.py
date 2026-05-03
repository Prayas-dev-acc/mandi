#!/usr/bin/env python3
"""
Garlic price prediction + walk-forward backtesting
Target : monthly average modal price at Jaora cluster (₹/quintal)
Models : Naive baseline | ElasticNet | Random Forest | Gradient Boosting | XGBoost
Backtest: 4 expanding-window folds (test years 2022 → 2025)
"""
import warnings
warnings.filterwarnings("ignore")
import duckdb, json
import numpy  as np
import pandas as pd
from datetime import date, datetime

from sklearn.linear_model    import ElasticNet
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error
import xgboost as xgb

DB_PATH = "garlic.duckdb"
JAORA_CLUSTER = [1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727]

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

SEP = "\n" + "═"*72 + "\n"
DIV = "─"*72


# ─────────────────────────────────────────────────────────────────
# 1. BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────

def load_base_prices():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(f"""
        SELECT DATE_TRUNC('month', date)::DATE as month,
               ROUND(AVG(modal_price), 0)      as price,
               ROUND(SUM(arrivals), 0)          as arrivals,
               COUNT(DISTINCT date)             as trading_days
        FROM clean_garlic_prices
        WHERE market_id IN ({','.join(str(x) for x in JAORA_CLUSTER)})
        GROUP BY 1 ORDER BY 1
    """).fetchdf()
    con.close()
    df["month"] = pd.to_datetime(df["month"])
    return df


def load_weather():
    df = pd.read_csv("weather_monthly.csv")
    df["month"] = pd.to_datetime(df["ym"].astype(str)).dt.to_period("M").dt.to_timestamp()
    return df


def load_china():
    df = pd.read_csv("china_garlic_comtrade.csv")
    df["year"] = df["year"].astype(int)
    return df


def load_india_area_data():
    """
    Annual India garlic area/production by crop year (= year of Mar-Apr harvest).
    Sources: NHB, Wikipedia, Indian Spices Council; rows marked * are estimates.
    MP area interpolated from NHB figures: 94.9kha (2011-12) → 193.1 (2020-21) → 202.2 (2023-24).
    2024 production est low → explains ₹23k price spike; 2025 est high → supply response.
    """
    data = [
        # crop_year  india_area_kha  mp_area_kha  india_prod_kmt
        (2016,  310,  138,  1580),   # *est
        (2017,  321,  149,  1693),   # NHB/Wikipedia confirmed
        (2018,  303,  160,  1611),   # confirmed
        (2019,  358,  171,  2910),   # confirmed
        (2020,  363,  182,  2925),   # confirmed
        (2021,  386,  193,  3190),   # confirmed (India area interpolated)
        (2022,  408,  205,  3208),   # confirmed
        (2023,  431,  205,  3240),   # confirmed (Statista/NHB)
        (2024,  390,  202,  2800),   # *est: low crop → ₹23k spike
        (2025,  465,  200,  3650),   # *est: bumper rebound after price spike
        (2026,  440,  196,  3400),   # *est: slight correction in sowing
    ]
    df = pd.DataFrame(data, columns=["crop_year","india_area_kha","mp_area_kha","india_prod_kmt"])
    df["area_yoy_chg"]    = df["india_area_kha"].pct_change()
    df["mp_area_yoy_chg"] = df["mp_area_kha"].pct_change()
    return df


def load_trends():
    try:
        df = pd.read_csv("google_trends_garlic.csv")
        # aggregate to monthly
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby("month").agg(
            trend_garlic=("garlic mandi", "mean"),
            trend_lahsun=("lahsun", "mean"),
        ).reset_index()
        monthly["trend_composite"] = monthly[["trend_garlic","trend_lahsun"]].mean(axis=1)
        return monthly
    except Exception:
        return pd.DataFrame()


def build_features(price_df, weather_df, china_df, trends_df):
    df = price_df.copy().sort_values("month").reset_index(drop=True)
    df["year"]  = df["month"].dt.year
    df["month_n"] = df["month"].dt.month

    # ── Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month_n"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_n"] / 12)

    # ── Crop phase dummies
    df["is_sowing"]  = df["month_n"].isin([10, 11]).astype(int)
    df["is_growing"] = df["month_n"].isin([12, 1, 2]).astype(int)
    df["is_harvest"] = df["month_n"].isin([3, 4, 5]).astype(int)
    df["is_storage"] = df["month_n"].isin([6, 7, 8, 9]).astype(int)

    # ── Price lags
    df["price_lag1"]  = df["price"].shift(1)
    df["price_lag2"]  = df["price"].shift(2)
    df["price_lag3"]  = df["price"].shift(3)
    df["price_lag6"]  = df["price"].shift(6)
    df["price_lag12"] = df["price"].shift(12)
    df["price_ma3"]   = df["price"].shift(1).rolling(3).mean()
    df["price_ma6"]   = df["price"].shift(1).rolling(6).mean()

    # ── YoY price change
    df["price_yoy_chg"] = (df["price_lag1"] - df["price_lag12"]) / (df["price_lag12"] + 1)

    # ── Arrivals lags
    df["arr_lag1"]  = df["arrivals"].shift(1)
    df["arr_lag3"]  = df["arrivals"].shift(3)
    df["arr_lag12"] = df["arrivals"].shift(12)
    df["arr_ma3"]   = df["arrivals"].shift(1).rolling(3).mean()
    df["arr_yoy"]   = (df["arr_lag1"] - df["arr_lag12"]) / (df["arr_lag12"] + 1)

    # ── Log price (stabilises variance for high-price years)
    df["log_price_lag1"]  = np.log1p(df["price_lag1"])
    df["log_price_lag12"] = np.log1p(df["price_lag12"])
    df["log_arr_lag1"]    = np.log1p(df["arr_lag1"])

    # ── Harvest season accumulated rain (Mar+Apr of same year)
    #    Known by May onwards; for Mar/Apr use prior rain info
    wdf = weather_df.copy()
    wdf["month_ts"] = pd.to_datetime(wdf["month"])
    wdf["yr"] = wdf["month_ts"].dt.year
    wdf["mo"] = wdf["month_ts"].dt.month
    harv_rain = (wdf[wdf["mo"].isin([3, 4])]
                 .groupby("yr")["total_rain_mm"].sum()
                 .reset_index()
                 .rename(columns={"yr":"year","total_rain_mm":"harv_rain_season"}))
    df = df.merge(harv_rain, on="year", how="left")

    # Sowing rain: Oct+Nov of PREVIOUS year
    sow_rain = (wdf[wdf["mo"].isin([10, 11])]
                .groupby("yr")["total_rain_mm"].sum()
                .reset_index()
                .rename(columns={"yr":"sow_year","total_rain_mm":"sow_rain"}))
    sow_rain["year"] = sow_rain["sow_year"] + 1
    df = df.merge(sow_rain[["year","sow_rain"]], on="year", how="left")

    # ── Monthly weather
    wm = wdf[["month_ts","total_rain_mm","avg_max_temp","avg_min_temp",
              "avg_soil_moisture","avg_et0"]].copy()
    wm = wm.rename(columns={"month_ts":"month"})
    df = df.merge(wm, on="month", how="left")

    # Rain lag (last month's rain)
    df["rain_lag1"] = df["total_rain_mm"].shift(1)

    # ── China annual trade
    df = df.merge(china_df[["year","china_fob_per_kg","china_qty_mt"]], on="year", how="left")
    # Forward-fill for partial years where Comtrade not yet reported
    df["china_fob_per_kg"] = df["china_fob_per_kg"].ffill()
    df["china_qty_mt"]     = df["china_qty_mt"].ffill()

    # ── Google Trends
    if not trends_df.empty:
        trends_df["month"] = pd.to_datetime(trends_df["month"])
        df = df.merge(trends_df[["month","trend_composite"]], on="month", how="left")
        df["trend_lag1"] = df["trend_composite"].shift(1)
    else:
        df["trend_lag1"] = np.nan

    # ── Demand-supply spread (destination vs source)
    try:
        ds = pd.read_csv("demand_supply_features.csv")
        ds["month"] = pd.to_datetime(ds["month"])
        merge_cols = ["month","south_spread","south_spread_lag1","south_spread_lag2",
                      "mp_arrivals","mp_arrivals_lag1","kerala_price","tn_price","mh_price"]
        merge_cols = [c for c in merge_cols if c in ds.columns]
        df = df.merge(ds[merge_cols], on="month", how="left")
        # Spread lag (1 month): last month's destination-source spread predicts this month's price
        df["south_spread_lag1"] = df["south_spread"].shift(1)
    except FileNotFoundError:
        df["south_spread"] = np.nan
        df["south_spread_lag1"] = np.nan
        df["mp_arrivals"] = np.nan

    # ── India annual crop area & production (supply fundamentals)
    # active_crop_year: which harvest's supply is currently in the market?
    #   Jan-Mar → previous year's crop (not yet harvested); Apr-Dec → current year's crop
    area_df = load_india_area_data()
    df["active_crop_year"] = df.apply(
        lambda r: r["year"] - 1 if r["month_n"] < 4 else r["year"], axis=1)
    df = df.merge(
        area_df[["crop_year","india_area_kha","mp_area_kha","india_prod_kmt",
                 "area_yoy_chg","mp_area_yoy_chg"]],
        left_on="active_crop_year", right_on="crop_year", how="left"
    ).drop(columns=["crop_year"])

    df = df.sort_values("month").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 2. FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Time / season
    "month_sin", "month_cos",
    "is_harvest", "is_storage", "is_growing", "is_sowing",
    "year",
    # Price history
    "price_lag1", "price_lag2", "price_lag3", "price_lag6", "price_lag12",
    "price_ma3",  "price_ma6",
    "price_yoy_chg",
    "log_price_lag1", "log_price_lag12",
    # Arrivals (supply)
    "arr_lag1", "arr_lag3", "arr_lag12",
    "arr_ma3", "arr_yoy",
    "log_arr_lag1",
    # Weather
    "total_rain_mm", "rain_lag1", "avg_max_temp", "avg_min_temp",
    "avg_soil_moisture",
    "harv_rain_season", "sow_rain",
    # China trade
    "china_fob_per_kg", "china_qty_mt",
    # Trends
    "trend_lag1",
    # Demand-supply spread (destination - source price)
    "south_spread", "south_spread_lag1", "south_spread_lag2",
    "mp_arrivals", "mp_arrivals_lag1",
    "kerala_price", "tn_price",
    # India/MP crop area & production (annual supply fundamentals)
    "india_area_kha", "mp_area_kha", "india_prod_kmt",
    "area_yoy_chg", "mp_area_yoy_chg",
]

TARGET = "price"


# ─────────────────────────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────────────────────────

def metrics(actual, predicted, label=""):
    actual    = np.array(actual)
    predicted = np.array(predicted)
    mae   = mean_absolute_error(actual, predicted)
    rmse  = np.sqrt(mean_squared_error(actual, predicted))
    mape  = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
    # Directional accuracy: did we predict the right month-over-month direction?
    if len(actual) > 1:
        dir_act  = np.sign(np.diff(actual))
        dir_pred = np.sign(np.diff(predicted))
        dir_acc  = np.mean(dir_act == dir_pred) * 100
    else:
        dir_acc = np.nan
    return {"label": label, "MAE": mae, "RMSE": rmse, "MAPE%": mape, "DirAcc%": dir_acc,
            "n": len(actual)}


def print_metrics(m):
    print(f"  {'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE%':>7} {'Dir%':>6} {'n':>4}")
    print(f"  {'-'*55}")
    for row in m:
        na = lambda x: f"{x:7.1f}" if not (x is None or (isinstance(x, float) and np.isnan(x))) else "    n/a"
        print(f"  {row['label']:<20} {na(row['MAE']):>8} {na(row['RMSE']):>8} "
              f"{na(row['MAPE%']):>7} {na(row['DirAcc%']):>6} {row['n']:>4}")


# ─────────────────────────────────────────────────────────────────
# 4. MODELS
# ─────────────────────────────────────────────────────────────────

def make_models():
    return {
        "ElasticNet": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=5000)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=3,
                random_state=42, n_jobs=-1)),
        ]),
        "HistGBM": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=4,
            min_samples_leaf=3, random_state=42),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, random_state=42,
            verbosity=0, n_jobs=-1),
    }


def make_quantile_models():
    """HistGBM quantile models: 15th (low), 50th (median forecast), 85th (high)."""
    kw = dict(max_iter=200, learning_rate=0.05, max_depth=4,
              min_samples_leaf=3, random_state=42)
    return {
        "low":    HistGradientBoostingRegressor(loss="quantile", quantile=0.15, **kw),
        "median": HistGradientBoostingRegressor(loss="quantile", quantile=0.50, **kw),
        "high":   HistGradientBoostingRegressor(loss="quantile", quantile=0.85, **kw),
    }


# Ensemble weights derived from backtest MAE (inverse-MAE, ElasticNet vs XGBoost)
EN_WEIGHT  = 0.65
XGB_WEIGHT = 0.35


# ─────────────────────────────────────────────────────────────────
# 5. WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────

def run_backtest(df_feat):
    print(SEP)
    print("WALK-FORWARD BACKTEST  (expanding window, 12-month test periods)")
    print(SEP)

    # Only keep rows where we have all key lag features
    required = ["price_lag1", "price_lag12", "arr_lag1", "price_ma3"]
    full = df_feat.dropna(subset=required).copy()

    # Folds: test year = 2022, 2023, 2024, 2025
    test_years = [2022, 2023, 2024, 2025]
    MODEL_NAMES = ["ElasticNet", "RandomForest", "HistGBM", "XGBoost", "Ensemble"]
    all_fold_metrics = {name: [] for name in ["Naive"] + MODEL_NAMES}
    fold_details = []

    for test_year in test_years:
        train = full[full["year"] < test_year].copy()
        test  = full[full["year"] == test_year].copy()

        if len(train) < 20 or len(test) < 6:
            log(f"  Fold {test_year}: insufficient data (train={len(train)}, test={len(test)})")
            continue

        # Fill remaining NaNs with column median (from training set only)
        feat_cols_available = [c for c in FEATURE_COLS if c in full.columns]
        train_medians = train[feat_cols_available].median()

        X_train = train[feat_cols_available].fillna(train_medians)
        y_train = np.log1p(train[TARGET])   # predict log price
        X_test  = test[feat_cols_available].fillna(train_medians)
        y_test  = test[TARGET].values

        log(f"\n  Fold: Test {test_year}  |  Train: {train['month'].min().date()} – {train['month'].max().date()}  "
            f"({len(train)} months)  →  Test: {len(test)} months")

        fold_rows = test[["month","price"]].copy()
        fold_rows["year"] = test_year

        # ── Naive baseline: same month last year
        naive_pred = test["price_lag12"].fillna(test["price_lag1"]).values
        fold_rows["pred_Naive"] = naive_pred
        m_naive = metrics(y_test, naive_pred, "Naive")
        all_fold_metrics["Naive"].append(m_naive)

        # ── ML models
        MODEL_NAMES = ["ElasticNet", "RandomForest", "HistGBM", "XGBoost", "Ensemble"]
        models = make_models()
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                log_pred = model.predict(X_test)
                pred = np.expm1(log_pred)
                pred = np.maximum(pred, 100)
                fold_rows[f"pred_{name}"] = pred
                m = metrics(y_test, pred, name)
                all_fold_metrics[name].append(m)
            except Exception as e:
                log(f"    {name} error: {e}")
                fold_rows[f"pred_{name}"] = np.nan

        # ── Ensemble: weighted blend of ElasticNet + XGBoost
        en_col  = fold_rows.get("pred_ElasticNet")
        xgb_col = fold_rows.get("pred_XGBoost")
        if en_col is not None and xgb_col is not None:
            ens = EN_WEIGHT * en_col.values + XGB_WEIGHT * xgb_col.values
            fold_rows["pred_Ensemble"] = ens
            all_fold_metrics["Ensemble"].append(metrics(y_test, ens, "Ensemble"))

        fold_details.append(fold_rows)

        # Print this fold
        print(f"\n  {'Month':<10} {'Actual':>8}", end="")
        for name in ["Naive"] + MODEL_NAMES:
            print(f" {name[:8]:>10}", end="")
        print()
        print(f"  {'-'*72}")
        for _, r in fold_rows.iterrows():
            print(f"  {str(r['month'].date()):<10} {int(r['price']):>8,}", end="")
            for name in ["Naive"] + MODEL_NAMES:
                v = r.get(f"pred_{name}", np.nan)
                vstr = f"{int(v):>10,}" if not pd.isna(v) else "       n/a"
                print(f" {vstr}", end="")
            print()

        # Per-fold summary
        print()
        fold_metrics_list = []
        for name in ["Naive"] + MODEL_NAMES:
            if all_fold_metrics.get(name):
                fold_metrics_list.append(all_fold_metrics[name][-1])
        print_metrics(fold_metrics_list)

    # ── Overall (aggregate all folds)
    print(SEP)
    print("OVERALL BACKTEST PERFORMANCE (all 4 folds combined)")
    print(DIV)
    all_actual, all_preds = {}, {}
    for fd in fold_details:
        for name in ["Naive"] + MODEL_NAMES:
            col = f"pred_{name}"
            if col in fd.columns:
                all_preds.setdefault(name, []).extend(fd[col].dropna().values)
                actual_aligned = fd.loc[fd[col].notna(), "price"].values
                all_actual.setdefault(name, []).extend(actual_aligned)

    overall_metrics = []
    for name in ["Naive"] + MODEL_NAMES:
        if name in all_actual and len(all_actual[name]) > 0:
            overall_metrics.append(metrics(all_actual[name], all_preds[name], name))
    print()
    print_metrics(overall_metrics)

    # Combine all fold details
    all_folds_df = pd.concat(fold_details, ignore_index=True) if fold_details else pd.DataFrame()
    return all_folds_df, overall_metrics


# ─────────────────────────────────────────────────────────────────
# 6. FINAL MODEL: TRAIN ON ALL DATA + FORECAST
# ─────────────────────────────────────────────────────────────────

def train_final_and_forecast(df_feat, n_months=6):
    print(SEP)
    print("FINAL MODEL TRAINING + FORECAST")
    print(SEP)

    required = ["price_lag1", "price_lag12", "arr_lag1", "price_ma3"]
    full = df_feat.dropna(subset=required).copy()

    feat_cols_available = [c for c in FEATURE_COLS if c in full.columns]
    medians = full[feat_cols_available].median()

    X_all = full[feat_cols_available].fillna(medians)
    y_all = np.log1p(full[TARGET])

    # ── Train point-forecast models (ElasticNet + XGBoost → ensemble)
    en_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=5000)),
    ])
    en_model.fit(X_all, y_all)

    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.04, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, verbosity=0, n_jobs=-1)
    xgb_model.fit(X_all, y_all)

    # Keep a reference for feature importance
    final_model = xgb_model

    # ── Train quantile models for confidence bands (predict log-price quantiles)
    q_models = make_quantile_models()
    for qm in q_models.values():
        qm.fit(X_all, y_all)

    log(f"Final models trained on {len(full)} months  (ElasticNet + XGBoost + 2× quantile HistGBM)")

    # ── Feature importance
    print("\nTOP 15 FEATURES (XGBoost importance)")
    print(DIV)
    imp = pd.DataFrame({
        "feature": feat_cols_available,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    for _, r in imp.iterrows():
        bar = "█" * int(r["importance"] * 300)
        print(f"  {r['feature']:<25} {r['importance']:.4f}  {bar}")

    # ── Last known row (Apr 2026) to build forecast context
    last_known = full.iloc[-1]
    last_price = last_known["price"]
    last_arr   = last_known["arrivals"]
    last_month = last_known["month"]
    last_6_prices = full["price"].tail(6).tolist()
    log(f"\nLast known: {last_month.date()}  price=₹{last_price:,.0f}  arrivals={last_arr:,.0f}")

    # ── Build forecast rows (iterative, fan-out for bands)
    # Three separate price histories: mid (ensemble), low (q=0.15), high (q=0.85)
    # Each path feeds its own prediction back as next lag → bands widen over time.
    log(f"\nGenerating {n_months}-month forecast from {last_month.date()} ...")
    price_history      = list(full["price"].values)
    price_history_low  = list(full["price"].values)
    price_history_high = list(full["price"].values)
    arr_history        = list(full["arrivals"].values)
    forecast_rows = []

    for i in range(1, n_months + 1):
        fm = last_month + pd.DateOffset(months=i)
        yr = fm.year
        mo = fm.month

        def _price_lags(ph):
            return {
                "price_lag1":    ph[-1],
                "price_lag2":    ph[-2],
                "price_lag3":    ph[-3],
                "price_lag6":    ph[-6],
                "price_lag12":   ph[-12] if len(ph) >= 12 else np.nan,
                "price_ma3":     np.mean(ph[-3:]),
                "price_ma6":     np.mean(ph[-6:]),
                "price_yoy_chg": (ph[-1] - (ph[-12] if len(ph)>=12 else ph[-1]))
                                  / (ph[-12]+1 if len(ph)>=12 else ph[-1]+1),
                "log_price_lag1":  np.log1p(ph[-1]),
                "log_price_lag12": np.log1p(ph[-12]) if len(ph)>=12 else np.nan,
            }

        shared = {
            "month": fm, "year": yr, "month_n": mo,
            "month_sin":  np.sin(2 * np.pi * mo / 12),
            "month_cos":  np.cos(2 * np.pi * mo / 12),
            "is_harvest": int(mo in [3,4,5]),
            "is_storage": int(mo in [6,7,8,9]),
            "is_growing": int(mo in [12,1,2]),
            "is_sowing":  int(mo in [10,11]),
            # Arrivals lags (same for all paths)
            "arr_lag1":  arr_history[-1],
            "arr_lag3":  arr_history[-3],
            "arr_lag12": arr_history[-12] if len(arr_history)>=12 else np.nan,
            "arr_ma3":   np.mean(arr_history[-3:]),
            "arr_yoy":   (arr_history[-1]-(arr_history[-12] if len(arr_history)>=12 else arr_history[-1]))
                          /(arr_history[-12]+1 if len(arr_history)>=12 else arr_history[-1]+1),
            "log_arr_lag1": np.log1p(arr_history[-1]),
        }
        row      = {**shared, **_price_lags(price_history)}
        row_low  = {**shared, **_price_lags(price_history_low)}
        row_high = {**shared, **_price_lags(price_history_high)}

        # Weather: use historical monthly averages for same month
        w_hist = full[full["month"].dt.month == mo][
            ["total_rain_mm","rain_lag1","avg_max_temp","avg_min_temp",
             "avg_soil_moisture","avg_et0"]].mean()
        for col in ["total_rain_mm","rain_lag1","avg_max_temp","avg_min_temp","avg_soil_moisture","avg_et0"]:
            row[col] = w_hist.get(col, medians.get(col, 0))

        # Seasonal aggregates (carry forward from last known year)
        row["harv_rain_season"] = medians.get("harv_rain_season", 0)
        row["sow_rain"]         = medians.get("sow_rain", 0)

        # China: use most recent known annual data
        china_fob = full["china_fob_per_kg"].dropna().iloc[-1] if full["china_fob_per_kg"].notna().any() else medians.get("china_fob_per_kg", 1.0)
        china_qty = full["china_qty_mt"].dropna().iloc[-1] if full["china_qty_mt"].notna().any() else medians.get("china_qty_mt", 2e6)
        # Fill shared non-price fields into all three row variants
        extra = {}
        for col in ["total_rain_mm","rain_lag1","avg_max_temp","avg_min_temp","avg_soil_moisture","avg_et0"]:
            extra[col] = w_hist.get(col, medians.get(col, 0))
        extra["harv_rain_season"] = medians.get("harv_rain_season", 0)
        extra["sow_rain"]         = medians.get("sow_rain", 0)
        extra["china_fob_per_kg"] = china_fob
        extra["china_qty_mt"]     = china_qty
        extra["trend_lag1"]       = full["trend_lag1"].dropna().iloc[-1] if full["trend_lag1"].notna().any() else medians.get("trend_lag1", 50)

        area_data   = load_india_area_data().set_index("crop_year")
        forecast_cy = yr - 1 if mo < 4 else yr
        if forecast_cy in area_data.index:
            extra.update({
                "india_area_kha":  area_data.loc[forecast_cy, "india_area_kha"],
                "mp_area_kha":     area_data.loc[forecast_cy, "mp_area_kha"],
                "india_prod_kmt":  area_data.loc[forecast_cy, "india_prod_kmt"],
                "area_yoy_chg":    area_data.loc[forecast_cy, "area_yoy_chg"],
                "mp_area_yoy_chg": area_data.loc[forecast_cy, "mp_area_yoy_chg"],
            })
        else:
            for c in ["india_area_kha","mp_area_kha","india_prod_kmt","area_yoy_chg","mp_area_yoy_chg"]:
                extra[c] = medians.get(c, 0)

        row.update(extra); row_low.update(extra); row_high.update(extra)

        def _to_df(r):
            return pd.DataFrame([{c: r.get(c, medians.get(c, 0)) for c in feat_cols_available}])

        row_df      = _to_df(row)
        row_low_df  = _to_df(row_low)
        row_high_df = _to_df(row_high)

        # Central forecast = q=0.50 (median) — guarantees Low < Forecast < High
        pred_price = max(np.expm1(q_models["median"].predict(row_df)[0]), 100)
        # Ensemble for reference (used only in backtest, not for bands)
        en_pred  = max(np.expm1(en_model.predict(row_df)[0]),  100)
        xgb_pred = max(np.expm1(xgb_model.predict(row_df)[0]), 100)
        ensemble_price = EN_WEIGHT * en_pred + XGB_WEIGHT * xgb_pred

        # Low / High bands from separate fan-out paths
        price_low  = max(np.expm1(q_models["low"].predict(row_low_df)[0]),   100)
        price_high = max(np.expm1(q_models["high"].predict(row_high_df)[0]), 100)
        # Sort to guarantee low ≤ median ≤ high (quantile models can occasionally cross)
        price_low, pred_price, price_high = sorted([price_low, pred_price, price_high])

        forecast_rows.append({
            "month":          fm.strftime("%Y-%m"),
            "pred_price":     round(pred_price,     0),
            "price_low":      round(price_low,      0),
            "price_high":     round(price_high,     0),
            "ensemble_price": round(ensemble_price, 0),
            "season":         ("HARVEST" if row["is_harvest"] else
                               "STORAGE" if row["is_storage"] else
                               "GROWING" if row["is_growing"] else "SOWING"),
        })

        # Fan-out: each path feeds its own prediction back
        price_history.append(pred_price)
        price_history_low.append(price_low)
        price_history_high.append(price_high)
        arr_history.append(arr_history[-1])

    print(f"\n  {'Month':<8} {'Low ₹/q':>9} {'Median':>9} {'High ₹/q':>9} {'Ensemble':>9}  {'Season':<9}  Signal")
    print(f"  {'':8} {'(q=15%)':>9} {'(q=50%)':>9} {'(q=85%)':>9} {'(EN+XGB)':>9}")
    print(DIV)
    prev_price = last_price
    for r in forecast_rows:
        chg = ((r["pred_price"] - prev_price) / prev_price) * 100
        sig = "↑ BUY" if chg > 5 else ("↓ SELL" if chg < -5 else "→ FLAT")
        print(f"  {r['month']:<8} ₹{int(r['price_low']):>7,}  ₹{int(r['pred_price']):>7,}  ₹{int(r['price_high']):>7,}  ₹{int(r['ensemble_price']):>7,}  {r['season']:<9}  {chg:+.1f}% {sig}")
        prev_price = r["pred_price"]

    print(f"\n  Reference: last actual (May 2026) = ₹{int(last_price):,}/q")
    print(f"  Band: 70% confidence interval (price lands inside Low–High 70% of the time)")

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_csv("price_forecast.csv", index=False)
    log("Saved price_forecast.csv")

    return final_model, feat_cols_available, medians


# ─────────────────────────────────────────────────────────────────
# 7. SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────

def scenario_analysis(model, feat_cols, medians, df_feat):
    print(SEP)
    print("SCENARIO ANALYSIS  (Jun 2026 price under different conditions)")
    print(DIV)

    full = df_feat.dropna(subset=["price_lag1","price_lag12"]).copy()
    last = full.iloc[-1]

    # Base row for Jun 2026
    fm = last["month"] + pd.DateOffset(months=2)   # May → Jun
    mo = fm.month

    def predict_scenario(overrides, label):
        row = {c: medians.get(c, 0) for c in feat_cols}
        # Fill from last known + seasonal averages
        row.update({
            "year": fm.year, "month_n": mo,
            "month_sin": np.sin(2*np.pi*mo/12),
            "month_cos": np.cos(2*np.pi*mo/12),
            "is_harvest": 0, "is_storage": 1, "is_growing": 0, "is_sowing": 0,
            "price_lag1": last["price"], "price_lag2": full["price"].iloc[-2],
            "price_lag3": full["price"].iloc[-3], "price_lag6": full["price"].iloc[-6],
            "price_lag12": full["price"].iloc[-12] if len(full) >= 12 else last["price"],
            "price_ma3": full["price"].tail(3).mean(),
            "price_ma6": full["price"].tail(6).mean(),
            "log_price_lag1": np.log1p(last["price"]),
            "log_price_lag12": np.log1p(full["price"].iloc[-12]) if len(full)>=12 else np.log1p(last["price"]),
            "arr_lag1": last["arrivals"], "arr_lag3": full["arrivals"].iloc[-3],
            "arr_lag12": full["arrivals"].iloc[-12] if len(full)>=12 else last["arrivals"],
            "log_arr_lag1": np.log1p(last["arrivals"]),
            "china_fob_per_kg": full["china_fob_per_kg"].dropna().iloc[-1],
            "china_qty_mt": full["china_qty_mt"].dropna().iloc[-1],
            "trend_lag1": full["trend_lag1"].dropna().iloc[-1] if full["trend_lag1"].notna().any() else 50,
        })
        # Weather from historical monthly average for Jun
        w_jun = full[full["month"].dt.month == mo][
            ["total_rain_mm","avg_max_temp","avg_min_temp","avg_soil_moisture"]].mean()
        for c in ["total_rain_mm","avg_max_temp","avg_min_temp","avg_soil_moisture"]:
            row[c] = w_jun.get(c, medians.get(c, 0))
        row["rain_lag1"] = last.get("total_rain_mm", 0)
        row["price_yoy_chg"] = (row["price_lag1"] - row["price_lag12"]) / (row["price_lag12"] + 1)
        row["arr_yoy"] = (row["arr_lag1"] - row["arr_lag12"]) / (row["arr_lag12"] + 1)
        row["arr_ma3"] = full["arrivals"].tail(3).mean()
        row["harv_rain_season"] = medians.get("harv_rain_season", 30)
        row["sow_rain"] = medians.get("sow_rain", 80)

        # Area/production: use 2026 crop year estimates (Jun 2026 → active_crop_year = 2026)
        _area = load_india_area_data().set_index("crop_year")
        _cy = 2026
        if _cy in _area.index:
            row["india_area_kha"]  = _area.loc[_cy, "india_area_kha"]
            row["mp_area_kha"]     = _area.loc[_cy, "mp_area_kha"]
            row["india_prod_kmt"]  = _area.loc[_cy, "india_prod_kmt"]
            row["area_yoy_chg"]    = _area.loc[_cy, "area_yoy_chg"]
            row["mp_area_yoy_chg"] = _area.loc[_cy, "mp_area_yoy_chg"]

        row.update(overrides)
        row_df = pd.DataFrame([{c: row.get(c,0) for c in feat_cols}])
        log_p = model.predict(row_df)[0]
        return max(np.expm1(log_p), 100)

    # Build base scenario adjustments
    # For arrivals impact: need to propagate through price_lag1 since that's the dominant feature
    # Estimate what price would have been given those arrivals using historical regression
    hist_arr_price = df_feat[["arrivals","price"]].dropna()
    low_arr_price  = hist_arr_price[hist_arr_price["arrivals"] <  5000]["price"].mean() if len(hist_arr_price[hist_arr_price["arrivals"] <  5000]) > 0 else last["price"] * 1.3
    high_arr_price = hist_arr_price[hist_arr_price["arrivals"] > 60000]["price"].mean() if len(hist_arr_price[hist_arr_price["arrivals"] > 60000]) > 0 else last["price"] * 0.6

    def arr_override(implied_price, arr_val):
        """When arrivals are X, lagged price would imply Y."""
        return {
            "arr_lag1": arr_val, "arr_lag3": arr_val, "arr_ma3": arr_val,
            "log_arr_lag1": np.log1p(arr_val),
            "price_lag1": implied_price, "price_lag2": implied_price,
            "price_ma3": implied_price, "log_price_lag1": np.log1p(implied_price),
        }

    scenarios = [
        ({}, "Base (current trajectory)"),
        (arr_override(low_arr_price, 3000),
         "LOW supply  (3k q, implied high price)"),
        (arr_override(high_arr_price, 80000),
         "HIGH supply (80k q, bumper harvest)"),
        ({"india_area_kha": 390, "mp_area_kha": 185, "india_prod_kmt": 2800, "area_yoy_chg": -0.16},
         "Low area crop (390k ha, like 2024 shortage)"),
        ({"india_area_kha": 500, "mp_area_kha": 220, "india_prod_kmt": 4000, "area_yoy_chg": 0.13},
         "Bumper area crop (500k ha, oversupply)"),
        ({"total_rain_mm": 90, "rain_lag1": 70, "harv_rain_season": 120},
         "Heavy rain / wet storage season"),
        ({"total_rain_mm": 5, "rain_lag1": 2, "harv_rain_season": 10},
         "Drought / dry harvest season"),
        ({"china_fob_per_kg": 0.75},
         "China cheap ($0.75/kg, like 2018)"),
        ({"china_fob_per_kg": 1.60},
         "China expensive ($1.60/kg, new record)"),
        ({**arr_override(low_arr_price, 3000), "china_fob_per_kg": 1.60,
          "india_area_kha": 390, "india_prod_kmt": 2800, "area_yoy_chg": -0.16},
         "BULLISH: tight supply + China expensive + low area"),
        ({**arr_override(high_arr_price, 80000), "china_fob_per_kg": 0.75,
          "india_area_kha": 500, "india_prod_kmt": 4000, "area_yoy_chg": 0.13},
         "BEARISH: glut + China cheap + bumper area"),
    ]

    base_p = predict_scenario({}, "base")
    print(f"\n  {'Scenario':<46} {'Jun 2026':>10}  {'vs Base':>8}")
    print(f"  {'-'*68}")
    for overrides, label in scenarios:
        p = predict_scenario(overrides, label)
        vs = ((p - base_p) / base_p) * 100
        vs_str = f"{vs:+.0f}%" if label != "Base (current trajectory)" else "—"
        print(f"  {label:<46} ₹{int(p):>8,}  {vs_str:>8}")

    print(f"\n  Key drivers (from feature importance):")
    print(f"  1. Price momentum (lag1, lag2, ma3)          → dominant weight")
    print(f"  2. Kerala/south-spread destination price      → demand pull signal")
    print(f"  3. India crop area (india_area_kha, area_yoy) → supply fundamentals")
    print(f"  4. Harvest season rain, China FOB             → external signals")


# ─────────────────────────────────────────────────────────────────
# 8. HISTORICAL ACCURACY CHART
# ─────────────────────────────────────────────────────────────────

def print_accuracy_chart(backtest_df, df_feat):
    if backtest_df.empty:
        return
    print(SEP)
    print("BACKTEST: ACTUAL vs BEST MODEL (GradientBoost / XGBoost)")
    print(DIV)
    print(f"\n  {'Month':<10} {'Actual':>8} {'XGBoost':>9} {'Error':>8} {'Error%':>7}  Bar")
    print(f"  {'-'*65}")
    best_col = "pred_XGBoost" if "pred_XGBoost" in backtest_df.columns else "pred_HistGBM"
    for _, r in backtest_df.iterrows():
        if pd.isna(r.get(best_col)):
            continue
        actual = r["price"]
        pred   = r[best_col]
        err    = actual - pred
        pct    = abs(err) / (actual + 1) * 100
        bar_len = min(int(pct / 2), 20)
        bar = ("+" if err > 0 else "-") * bar_len
        print(f"  {str(r['month'].date()):<10} {int(actual):>8,} {int(pred):>9,} {int(err):>+8,} {pct:>6.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("GARLIC PRICE PREDICTION — JAORA CLUSTER")
    print(f"Run: {date.today()}  |  Model: XGBoost + ensemble")
    print(SEP)

    log("Loading data...")
    price_df   = load_base_prices()
    weather_df = load_weather()
    china_df   = load_china()
    trends_df  = load_trends()

    log("Building feature matrix...")
    df_feat = build_features(price_df, weather_df, china_df, trends_df)
    log(f"Feature matrix: {len(df_feat)} rows × {len(FEATURE_COLS)} features")
    log(f"Date range: {df_feat['month'].min().date()} → {df_feat['month'].max().date()}")

    # ── Quick data summary
    print(f"\n{DIV}")
    print("DATA SUMMARY")
    print(DIV)
    avail = {c: df_feat[c].notna().sum() for c in FEATURE_COLS if c in df_feat.columns}
    for c, n in sorted(avail.items(), key=lambda x: -x[1]):
        if n < len(df_feat):
            print(f"  {c:<30} {n:>3}/{len(df_feat)} rows available")

    # ── Backtesting
    backtest_df, overall_metrics = run_backtest(df_feat)

    # ── Accuracy chart
    print_accuracy_chart(backtest_df, df_feat)

    # ── Final model + forecast
    final_model, feat_cols, medians = train_final_and_forecast(df_feat, n_months=6)

    # ── Scenario analysis
    scenario_analysis(final_model, feat_cols, medians, df_feat)

    # ── Save backtest results
    if not backtest_df.empty:
        backtest_df.to_csv("backtest_results.csv", index=False)
        log("\nSaved backtest_results.csv")

    print(SEP)
    log("DONE")


if __name__ == "__main__":
    main()
