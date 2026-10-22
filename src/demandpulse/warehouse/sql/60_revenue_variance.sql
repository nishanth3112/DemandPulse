-- Revenue variance decomposition per store, current window vs base window.
--   revenue = trading weeks x baskets per week x units per basket x revenue per unit
-- Additive LMDI-I (log-mean Divisia): with L = (R1 - R0) / ln(R1 / R0), each factor's effect is
--   L * ln(factor1 / factor0)
-- and the four effects sum EXACTLY to R1 - R0 (no residual, order-independent).
--   trading_weeks_effect  weeks traded differ (part-window openings or closures)
--   traffic_effect        baskets per trading week (footfall at the till)
--   basket_size_effect    units per basket
--   price_mix_effect      revenue per unit: the average price paid, which moves with both
--                         price changes and a shift in what is bought (mix); the data has
--                         no list prices or promotions so these cannot be separated
-- Logs are undefined at zero, so a store with no revenue in the base window contributes its
-- whole revenue as new_store_effect, and one with none in the current window contributes
-- minus its base revenue as closed_store_effect.
CREATE OR REPLACE TABLE mart_revenue_variance_store AS
WITH sw AS (
    SELECT w.comparison, w.period, s.store_code,
           count(*) AS weeks,
           CAST(sum(s.revenue) AS DOUBLE) AS revenue,
           CAST(sum(s.baskets) AS DOUBLE) AS baskets,
           CAST(sum(s.units) AS DOUBLE) AS units
    FROM fct_store_week s JOIN dim_comparison_week w USING (shop_week)
    GROUP BY w.comparison, w.period, s.store_code
), cur AS (SELECT * FROM sw WHERE period = 'current'),
   base AS (SELECT * FROM sw WHERE period = 'base'),
joined AS (
    SELECT
        coalesce(c.comparison, b.comparison) AS comparison,
        coalesce(c.store_code, b.store_code) AS store_code,
        coalesce(c.weeks, 0) AS cur_weeks, coalesce(b.weeks, 0) AS base_weeks,
        coalesce(c.revenue, 0) AS cur_rev, coalesce(b.revenue, 0) AS base_rev,
        coalesce(c.baskets, 0) AS cur_baskets, coalesce(b.baskets, 0) AS base_baskets,
        coalesce(c.units, 0) AS cur_units, coalesce(b.units, 0) AS base_units
    FROM cur c FULL OUTER JOIN base b
        ON c.comparison = b.comparison AND c.store_code = b.store_code
), calc AS (
    SELECT
        j.*,
        st.store_region,
        st.store_format,
        CASE
            WHEN j.base_rev = 0 THEN 'new'
            WHEN j.cur_rev = 0 THEN 'closed'
            ELSE 'continuing'
        END AS store_status,
        CASE
            WHEN j.base_rev > 0 AND j.cur_rev > 0 THEN
                CASE WHEN j.cur_rev = j.base_rev THEN j.base_rev
                     ELSE (j.cur_rev - j.base_rev) / ln(j.cur_rev / j.base_rev) END
        END AS log_mean
    FROM joined j JOIN dim_store st USING (store_code)
)
SELECT
    comparison,
    store_code,
    store_region,
    store_format,
    store_status,
    base_rev AS base_revenue,
    cur_rev AS current_revenue,
    cur_rev - base_rev AS revenue_change,
    -- Only continuing stores have defined logs; the CASE keeps ln() away from zero values.
    CASE WHEN store_status = 'continuing'
         THEN log_mean * ln(CAST(cur_weeks AS DOUBLE) / base_weeks) ELSE 0 END
        AS trading_weeks_effect,
    CASE WHEN store_status = 'continuing'
         THEN log_mean * ln((cur_baskets / cur_weeks) / (base_baskets / base_weeks)) ELSE 0 END
        AS traffic_effect,
    CASE WHEN store_status = 'continuing'
         THEN log_mean * ln((cur_units / cur_baskets) / (base_units / base_baskets)) ELSE 0 END
        AS basket_size_effect,
    CASE WHEN store_status = 'continuing'
         THEN log_mean * ln((cur_rev / cur_units) / (base_rev / base_units)) ELSE 0 END
        AS price_mix_effect,
    CASE WHEN store_status = 'new' THEN cur_rev ELSE 0 END AS new_store_effect,
    CASE WHEN store_status = 'closed' THEN -base_rev ELSE 0 END AS closed_store_effect
FROM calc;

-- Bridge from base to current revenue by region (and 'ALL' nationally): effects are additive,
-- so regional and national rows are plain sums of the store rows.
CREATE OR REPLACE TABLE mart_revenue_variance_region AS
SELECT
    comparison,
    coalesce(store_region, 'ALL') AS store_region,
    sum(base_revenue) AS base_revenue,
    sum(current_revenue) AS current_revenue,
    sum(revenue_change) AS revenue_change,
    sum(trading_weeks_effect) AS trading_weeks_effect,
    sum(traffic_effect) AS traffic_effect,
    sum(basket_size_effect) AS basket_size_effect,
    sum(price_mix_effect) AS price_mix_effect,
    sum(new_store_effect) AS new_store_effect,
    sum(closed_store_effect) AS closed_store_effect
FROM mart_revenue_variance_store
GROUP BY GROUPING SETS ((comparison, store_region), (comparison))
