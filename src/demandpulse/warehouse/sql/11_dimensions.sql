-- week_idx counts weeks from the first data week using dates, so it stays a true time index
-- even if a week is missing (as in the 5K sample). period_idx groups weeks into 4-week periods.
CREATE OR REPLACE TABLE dim_date AS
SELECT
    shop_week,
    week_start,
    week_end,
    CAST(shop_week // 100 AS INTEGER) AS fiscal_year,
    CAST(shop_week % 100 AS INTEGER) AS fiscal_week,
    CAST(datediff('day', min(week_start) OVER (), week_start) // 7 + 1 AS INTEGER) AS week_idx,
    CAST(datediff('day', min(week_start) OVER (), week_start) // 28 + 1 AS INTEGER) AS period_idx
FROM (SELECT DISTINCT shop_week, week_start, week_end FROM silver);

-- Comparison windows over week_idx. A window is dropped when its base would start before the
-- first data week. window_vs_year_ago compares like-for-like weeks one year apart (seasonally
-- clean); window_vs_prior compares to the immediately preceding window (seasonality included).
CREATE OR REPLACE TABLE dim_comparison AS
WITH m AS (SELECT max(week_idx) AS last_idx FROM dim_date)
SELECT * FROM (
    SELECT 'window_vs_prior' AS comparison,
           last_idx - $window_weeks + 1 AS current_from, last_idx AS current_to,
           last_idx - 2 * $window_weeks + 1 AS base_from, last_idx - $window_weeks AS base_to
    FROM m
    UNION ALL
    SELECT 'window_vs_year_ago',
           last_idx - $window_weeks + 1, last_idx,
           last_idx - $year_weeks - $window_weeks + 1, last_idx - $year_weeks
    FROM m
    UNION ALL
    SELECT 'year_vs_prior_year',
           last_idx - $year_weeks + 1, last_idx,
           last_idx - 2 * $year_weeks + 1, last_idx - $year_weeks
    FROM m
)
WHERE base_from >= 1;

CREATE OR REPLACE TABLE dim_store AS
WITH s AS (
    SELECT
        b.store_code,
        min(b.store_format) AS store_format,
        min(b.store_region) AS store_region,
        min(b.shop_week) AS first_week,
        max(b.shop_week) AS last_week,
        min(d.week_idx) AS first_week_idx,
        max(d.week_idx) AS last_week_idx,
        CAST(count(DISTINCT b.shop_week) AS INTEGER) AS weeks_active
    FROM fct_basket b JOIN dim_date d USING (shop_week)
    GROUP BY b.store_code
), m AS (SELECT max(week_idx) AS last_idx FROM dim_date)
SELECT
    s.*,
    CASE
        WHEN first_week_idx > 1 AND last_week_idx < last_idx THEN 'opened_and_closed'
        WHEN first_week_idx > 1 THEN 'opened_in_period'
        WHEN last_week_idx < last_idx THEN 'closed_in_period'
        ELSE 'open_throughout'
    END AS lifecycle
FROM s CROSS JOIN m;

CREATE OR REPLACE TABLE dim_customer AS
SELECT
    b.cust_code,
    min(b.seg_1) AS seg_1,
    min(b.seg_2) AS seg_2,
    min(b.shop_week) AS first_week,
    max(b.shop_week) AS last_week,
    min(d.week_idx) AS first_week_idx,
    max(d.week_idx) AS last_week_idx
FROM fct_basket b JOIN dim_date d USING (shop_week)
WHERE b.cust_code IS NOT NULL
GROUP BY b.cust_code;

-- Hierarchy levels nest strictly (verified on the data): 10 is the finest, 40 the coarsest.
-- Business labels used downstream: 40 division, 30 category, 20 sub-category, 10 class.
CREATE OR REPLACE TABLE dim_product AS
SELECT
    prod_code,
    min(prod_code_10) AS prod_code_10,
    min(prod_code_20) AS prod_code_20,
    min(prod_code_30) AS prod_code_30,
    min(prod_code_40) AS prod_code_40,
    min(shop_week) AS first_week,
    max(shop_week) AS last_week
FROM silver
GROUP BY prod_code;

-- Weeks belonging to each comparison window, so marts can join on shop_week.
CREATE OR REPLACE TABLE dim_comparison_week AS
SELECT c.comparison, 'current' AS period, d.shop_week, d.week_idx
FROM dim_comparison c JOIN dim_date d ON d.week_idx BETWEEN c.current_from AND c.current_to
UNION ALL
SELECT c.comparison, 'base' AS period, d.shop_week, d.week_idx
FROM dim_comparison c JOIN dim_date d ON d.week_idx BETWEEN c.base_from AND c.base_to
