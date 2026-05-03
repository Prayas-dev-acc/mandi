#!/usr/bin/env python3
"""
Comprehensive garlic market insights focused on Jaora mandi and
Mandsaur/Neemuch cluster. Garlic crop calendar for MP:
  - Sowing:    Oct–Nov
  - Growing:   Nov–Feb
  - Harvest:   Feb–Apr  (arrivals peak Mar–May)
  - Storage:   May–Sep  (old crop, prices rise)
  - New crop:  Oct–Nov  (price dip on anticipation)
"""
import duckdb, json
from datetime import date

con = duckdb.connect("garlic.duckdb", read_only=True)

# ── Focus markets ────────────────────────────────────────────────
JAORA_CLUSTER = [1085, 522, 182, 2336, 2088, 2111, 2662, 2082, 2727]  # Jaora, Mandsaur, Neemuch, Piplya, Sailana, Manasa, Sitmau, Badnagar, Javad
CLUSTER_NAMES = "Jaora, Mandsaur, Neemuch, Piplya, Sailana, Manasa, Sitmau, Badnagar, Javad"
CLUSTER_SQL   = ",".join(str(x) for x in JAORA_CLUSTER)

results = {}

# ────────────────────────────────────────────────────────────────
# 1. YEARLY PRICE TREND — cluster vs all-India
# ────────────────────────────────────────────────────────────────
results["1_yearly_trend"] = con.execute(f"""
SELECT
    YEAR(date) as year,
    ROUND(AVG(modal_price) FILTER (WHERE market_id IN ({CLUSTER_SQL})), 0) as cluster_avg_modal,
    ROUND(AVG(modal_price),0) as india_avg_modal,
    ROUND(AVG(modal_price) FILTER (WHERE market_id IN ({CLUSTER_SQL})) - AVG(modal_price), 0) as cluster_vs_india,
    ROUND(MIN(modal_price) FILTER (WHERE market_id IN ({CLUSTER_SQL})), 0) as cluster_min,
    ROUND(MAX(modal_price) FILTER (WHERE market_id IN ({CLUSTER_SQL})), 0) as cluster_max,
    SUM(arrivals) FILTER (WHERE market_id IN ({CLUSTER_SQL})) as cluster_total_arrivals_mt
FROM clean_garlic_prices
GROUP BY 1 ORDER BY 1
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 2. MONTHLY SEASONALITY — average price by month (all years, cluster)
# ────────────────────────────────────────────────────────────────
results["2_monthly_seasonality"] = con.execute(f"""
SELECT
    MONTH(date) as month,
    CASE MONTH(date)
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
    END as month_name,
    ROUND(AVG(modal_price), 0) as avg_modal,
    ROUND(MIN(modal_price), 0) as min_modal,
    ROUND(MAX(modal_price), 0) as max_modal,
    ROUND(AVG(arrivals), 2) as avg_daily_arrivals_mt,
    -- Phase tag
    CASE
        WHEN MONTH(date) IN (2,3,4,5) THEN 'HARVEST (sell carefully)'
        WHEN MONTH(date) IN (6,7,8,9) THEN 'STORAGE SEASON (prices rise)'
        WHEN MONTH(date) IN (10,11)   THEN 'SOWING (new crop anticipation)'
        WHEN MONTH(date) IN (12,1)    THEN 'GROWING (limited arrivals)'
    END as crop_phase
FROM clean_garlic_prices
WHERE market_id IN ({CLUSTER_SQL})
GROUP BY 1,2 ORDER BY 1
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 3. BEST MONTH TO SELL — by year (when did peak price occur?)
# ────────────────────────────────────────────────────────────────
results["3_best_sell_month_by_year"] = con.execute(f"""
WITH monthly AS (
    SELECT
        YEAR(date) as year,
        MONTH(date) as month,
        ROUND(AVG(modal_price), 0) as avg_modal
    FROM clean_garlic_prices
    WHERE market_id IN ({CLUSTER_SQL})
    GROUP BY 1,2
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY year ORDER BY avg_modal DESC) as rnk
    FROM monthly
)
SELECT year,
    CASE month
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
    END as best_month,
    avg_modal as peak_price
FROM ranked WHERE rnk=1 ORDER BY year
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 4. WORST MONTH TO SELL — harvest glut
# ────────────────────────────────────────────────────────────────
results["4_worst_sell_month_by_year"] = con.execute(f"""
WITH monthly AS (
    SELECT
        YEAR(date) as year,
        MONTH(date) as month,
        ROUND(AVG(modal_price), 0) as avg_modal
    FROM clean_garlic_prices
    WHERE market_id IN ({CLUSTER_SQL})
    GROUP BY 1,2
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY year ORDER BY avg_modal ASC) as rnk
    FROM monthly
)
SELECT year,
    CASE month
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
    END as worst_month,
    avg_modal as trough_price
FROM ranked WHERE rnk=1 ORDER BY year
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 5. PEAK-TO-TROUGH SPREAD PER YEAR (how much price swing in a year?)
# ────────────────────────────────────────────────────────────────
results["5_annual_price_swing"] = con.execute(f"""
SELECT
    YEAR(date) as year,
    ROUND(MIN(modal_price), 0) as year_low,
    ROUND(MAX(modal_price), 0) as year_high,
    ROUND(MAX(modal_price) - MIN(modal_price), 0) as swing,
    ROUND((MAX(modal_price) - MIN(modal_price)) / MIN(modal_price) * 100, 1) as swing_pct
FROM clean_garlic_prices
WHERE market_id IN ({CLUSTER_SQL})
GROUP BY 1 ORDER BY 1
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 6. JAORA vs NEEMUCH vs MANDSAUR — head-to-head price comparison
# ────────────────────────────────────────────────────────────────
results["6_market_comparison"] = con.execute("""
SELECT
    YEAR(date) as year,
    ROUND(AVG(modal_price) FILTER (WHERE market_id = 1085), 0) as Jaora,
    ROUND(AVG(modal_price) FILTER (WHERE market_id = 522),  0) as Mandsaur,
    ROUND(AVG(modal_price) FILTER (WHERE market_id = 182),  0) as Neemuch,
    ROUND(AVG(modal_price) FILTER (WHERE market_id = 2336), 0) as Piplya,
    ROUND(AVG(modal_price) FILTER (WHERE market_id = 2088), 0) as Sailana
FROM clean_garlic_prices
WHERE market_id IN (1085, 522, 182, 2336, 2088)
GROUP BY 1 ORDER BY 1
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 7. ARRIVALS vs PRICE CORRELATION (monthly, cluster)
# Does more supply = lower price?
# ────────────────────────────────────────────────────────────────
results["7_arrivals_price_correlation"] = con.execute(f"""
SELECT
    MONTH(date) as month,
    CASE MONTH(date)
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
    END as month_name,
    ROUND(AVG(arrivals), 2)    as avg_arrivals_mt,
    ROUND(AVG(modal_price), 0) as avg_price
FROM clean_garlic_prices
WHERE market_id IN ({CLUSTER_SQL})
GROUP BY 1,2 ORDER BY 1
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 8. GRADE/VARIETY PREMIUM — does FAQ fetch more than Local?
# ────────────────────────────────────────────────────────────────
results["8_grade_variety_premium"] = con.execute(f"""
SELECT
    grade, variety,
    COUNT(*) as days_reported,
    ROUND(AVG(modal_price), 0) as avg_price,
    ROUND(MIN(modal_price), 0) as min_price,
    ROUND(MAX(modal_price), 0) as max_price
FROM clean_garlic_prices
WHERE market_id IN ({CLUSTER_SQL})
GROUP BY 1,2
HAVING COUNT(*) > 30
ORDER BY avg_price DESC
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 9. RECENT PRICE TREND — last 90 days at Jaora
# ────────────────────────────────────────────────────────────────
results["9_jaora_last_90_days"] = con.execute("""
SELECT date, variety, grade,
    arrivals, modal_price, min_price, max_price
FROM clean_garlic_prices
WHERE market_id = 1085
  AND date >= CURRENT_DATE - INTERVAL 90 DAY
ORDER BY date DESC
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 10. PRICE CRASH ALERTS — worst 10 single-day prices at Jaora ever
# ────────────────────────────────────────────────────────────────
results["10_jaora_price_crashes"] = con.execute("""
SELECT date, variety, grade, arrivals, modal_price, min_price, max_price
FROM clean_garlic_prices
WHERE market_id = 1085
ORDER BY modal_price ASC LIMIT 10
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 11. PRICE PEAKS — best 10 single-day prices at Jaora ever
# ────────────────────────────────────────────────────────────────
results["11_jaora_price_peaks"] = con.execute("""
SELECT date, variety, grade, arrivals, modal_price, min_price, max_price
FROM clean_garlic_prices
WHERE market_id = 1085
ORDER BY modal_price DESC LIMIT 10
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 12. HARVEST TIMING SIGNAL — when do arrivals FIRST spike each year?
#     (proxy for harvest start date)
# ────────────────────────────────────────────────────────────────
results["12_harvest_start_by_year"] = con.execute(f"""
WITH daily AS (
    SELECT date, YEAR(date) as yr, MONTH(date) as mo,
           SUM(arrivals) as total_arr
    FROM clean_garlic_prices
    WHERE market_id IN ({CLUSTER_SQL})
      AND MONTH(date) BETWEEN 1 AND 6
    GROUP BY 1,2,3
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY yr ORDER BY total_arr DESC) as rnk
    FROM daily
)
SELECT yr as year, date as peak_arrival_date, ROUND(total_arr,1) as arrivals_mt
FROM ranked WHERE rnk=1 ORDER BY yr
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 13. MARKET PREMIUM — does Jaora consistently pay more or less
#     than state average?
# ────────────────────────────────────────────────────────────────
results["13_jaora_vs_mp_avg"] = con.execute("""
SELECT
    YEAR(date) as year,
    MONTH(date) as month,
    ROUND(AVG(modal_price) FILTER (WHERE market_id = 1085), 0) as jaora_price,
    ROUND(AVG(modal_price) FILTER (WHERE state_name = 'Madhya Pradesh'), 0) as mp_avg_price,
    ROUND(AVG(modal_price) FILTER (WHERE market_id = 1085) -
          AVG(modal_price) FILTER (WHERE state_name = 'Madhya Pradesh'), 0) as premium
FROM clean_garlic_prices
WHERE state_name = 'Madhya Pradesh'
GROUP BY 1,2
HAVING AVG(modal_price) FILTER (WHERE market_id = 1085) IS NOT NULL
ORDER BY 1,2
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 14. NATIONAL PRICE BENCHMARK — where does MP cluster rank?
# ────────────────────────────────────────────────────────────────
results["14_state_price_ranking"] = con.execute("""
SELECT
    state_name,
    ROUND(AVG(modal_price), 0) as avg_modal,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY modal_price), 0) as median_modal,
    COUNT(DISTINCT market_id) as markets,
    ROUND(SUM(arrivals), 0) as total_arrivals_mt
FROM clean_garlic_prices
GROUP BY 1 ORDER BY 2 DESC
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 15. ROLLING 30-DAY PRICE — last 2 years at Jaora (trend signal)
# ────────────────────────────────────────────────────────────────
results["15_jaora_rolling30"] = con.execute("""
SELECT
    date,
    modal_price,
    ROUND(AVG(modal_price) OVER (
        ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 0) as rolling_30d_avg
FROM clean_garlic_prices
WHERE market_id = 1085
  AND date >= CURRENT_DATE - INTERVAL 2 YEAR
ORDER BY date
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 16. STORAGE PREMIUM — how much does price rise from harvest (Apr)
#     to peak storage (Aug/Sep)?
# ────────────────────────────────────────────────────────────────
results["16_storage_premium_by_year"] = con.execute(f"""
WITH seasonal AS (
    SELECT
        YEAR(date) as year,
        ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (3,4)), 0) as harvest_price,
        ROUND(AVG(modal_price) FILTER (WHERE MONTH(date) IN (8,9)), 0) as storage_price
    FROM clean_garlic_prices
    WHERE market_id IN ({CLUSTER_SQL})
    GROUP BY 1
)
SELECT year, harvest_price, storage_price,
    ROUND(storage_price - harvest_price, 0) as storage_gain,
    ROUND((storage_price - harvest_price) / NULLIF(harvest_price, 0) * 100, 1) as gain_pct
FROM seasonal
WHERE harvest_price IS NOT NULL AND storage_price IS NOT NULL
ORDER BY year
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 17. PRICE VOLATILITY — std deviation by month (risk calendar)
# ────────────────────────────────────────────────────────────────
results["17_price_volatility_by_month"] = con.execute(f"""
SELECT
    MONTH(date) as month,
    CASE MONTH(date)
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
    END as month_name,
    ROUND(AVG(modal_price), 0) as avg_price,
    ROUND(STDDEV(modal_price), 0) as std_dev,
    ROUND(STDDEV(modal_price) / AVG(modal_price) * 100, 1) as coeff_variation_pct
FROM clean_garlic_prices
WHERE market_id IN ({CLUSTER_SQL})
GROUP BY 1,2 ORDER BY 1
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 18. INTER-MARKET ARBITRAGE — same day price differences
# ────────────────────────────────────────────────────────────────
results["18_arbitrage_opportunities"] = con.execute("""
WITH daily_markets AS (
    SELECT date,
        AVG(modal_price) FILTER (WHERE market_id = 1085) as jaora,
        AVG(modal_price) FILTER (WHERE market_id = 522)  as mandsaur,
        AVG(modal_price) FILTER (WHERE market_id = 182)  as neemuch
    FROM clean_garlic_prices
    WHERE market_id IN (1085, 522, 182)
    GROUP BY date
    HAVING jaora IS NOT NULL AND mandsaur IS NOT NULL AND neemuch IS NOT NULL
)
SELECT
    YEAR(date) as year,
    ROUND(AVG(ABS(jaora - mandsaur)), 0) as avg_jaora_mandsaur_diff,
    ROUND(AVG(ABS(jaora - neemuch)),  0) as avg_jaora_neemuch_diff,
    ROUND(AVG(ABS(mandsaur - neemuch)), 0) as avg_mandsaur_neemuch_diff,
    ROUND(MAX(ABS(jaora - neemuch)), 0) as max_jaora_neemuch_diff
FROM daily_markets
GROUP BY 1 ORDER BY 1
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 19. YEAR-OVER-YEAR CHANGE — was 2024 better than 2023?
# ────────────────────────────────────────────────────────────────
results["19_yoy_price_change"] = con.execute(f"""
WITH yearly AS (
    SELECT YEAR(date) as year, ROUND(AVG(modal_price),0) as avg_price
    FROM clean_garlic_prices WHERE market_id IN ({CLUSTER_SQL})
    GROUP BY 1
)
SELECT a.year,
    a.avg_price,
    b.avg_price as prev_year_price,
    ROUND(a.avg_price - b.avg_price, 0) as change,
    ROUND((a.avg_price - b.avg_price) / NULLIF(b.avg_price,0) * 100, 1) as change_pct
FROM yearly a LEFT JOIN yearly b ON a.year = b.year + 1
ORDER BY a.year
""").fetchdf()

# ────────────────────────────────────────────────────────────────
# 20. CURRENT SEASON SNAPSHOT — this year vs last 3 years same period
# ────────────────────────────────────────────────────────────────
results["20_current_season_vs_history"] = con.execute(f"""
SELECT
    YEAR(date) as year,
    MONTH(date) as month,
    ROUND(AVG(modal_price), 0) as avg_modal,
    ROUND(SUM(arrivals), 1) as total_arrivals_mt
FROM clean_garlic_prices
WHERE market_id IN ({CLUSTER_SQL})
  AND MONTH(date) IN (1,2,3,4,5)
  AND YEAR(date) >= 2022
GROUP BY 1,2 ORDER BY 2,1
""").fetchdf()

con.close()

# ── PRINT ALL INSIGHTS ───────────────────────────────────────────
SEP = "\n" + "═"*70 + "\n"

print(SEP)
print("GARLIC MARKET INSIGHTS — JAORA CLUSTER (Mandsaur & Neemuch)")
print(f"Cluster: {CLUSTER_NAMES}")
print(SEP)

labels = {
    "1_yearly_trend":            "1. YEARLY PRICE TREND — Cluster vs All-India (Rs./Quintal)",
    "2_monthly_seasonality":     "2. MONTHLY SEASONALITY — Avg price + crop phase",
    "3_best_sell_month_by_year": "3. BEST MONTH TO SELL each year (peak price month)",
    "4_worst_sell_month_by_year":"4. WORST MONTH TO SELL each year (trough price month)",
    "5_annual_price_swing":      "5. ANNUAL PRICE SWING — year high vs low",
    "6_market_comparison":       "6. MARKET HEAD-TO-HEAD — Jaora vs nearby mandis",
    "7_arrivals_price_correlation":"7. ARRIVALS vs PRICE — supply/demand by month",
    "8_grade_variety_premium":   "8. GRADE/VARIETY PREMIUM — what earns more?",
    "9_jaora_last_90_days":      "9. JAORA — last 90 days",
    "10_jaora_price_crashes":    "10. JAORA — worst price days ever (crashes)",
    "11_jaora_price_peaks":      "11. JAORA — best price days ever (peaks)",
    "12_harvest_start_by_year":  "12. HARVEST START SIGNAL — peak arrival day each year",
    "13_jaora_vs_mp_avg":        "13. JAORA vs MP STATE AVG — does Jaora pay premium?",
    "14_state_price_ranking":    "14. NATIONAL BENCHMARK — state price ranking",
    "15_jaora_rolling30":        "15. JAORA — rolling 30-day price (last 2 years)",
    "16_storage_premium_by_year":"16. STORAGE PREMIUM — harvest price vs Aug/Sep price",
    "17_price_volatility_by_month":"17. PRICE RISK CALENDAR — volatility by month",
    "18_arbitrage_opportunities":"18. INTER-MARKET ARBITRAGE — price gaps between mandis",
    "19_yoy_price_change":       "19. YEAR-OVER-YEAR PRICE CHANGE",
    "20_current_season_vs_history":"20. THIS HARVEST vs LAST 3 YEARS (Jan–May)",
}

for key, label in labels.items():
    print(f"\n{'─'*70}")
    print(label)
    print('─'*70)
    df = results[key]
    print(df.to_string(index=False))

# Save to JSON for later use
with open("insights_data.json", "w") as f:
    json.dump({k: v.to_dict(orient="records") for k, v in results.items()}, f, indent=2, default=str)
print(f"\n\nAll data saved to insights_data.json")
