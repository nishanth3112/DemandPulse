-- Store scorecard per comparison window. Peer group for indices is the store's format (a size
-- class: LS, MS, SS, XLS), ranks are within region. Revenue is normalised per trading week so
-- stores that traded only part of a window are not penalised.
-- Peer statistics use the median (robust to a few very large or small stores). Growth gaps and
-- z-scores use like-for-like stores only, so openings and closures do not distort the peer norm.
CREATE OR REPLACE TABLE mart_store_scorecard AS
WITH sw AS (
    SELECT w.comparison, w.period, s.store_code,
           count(*) AS weeks_traded,
           CAST(sum(s.revenue) AS DOUBLE) AS revenue,
           sum(s.baskets) AS baskets,
           sum(s.units) AS units,
           sum(s.loyalty_baskets) AS loyalty_baskets
    FROM fct_store_week s JOIN dim_comparison_week w USING (shop_week)
    GROUP BY w.comparison, w.period, s.store_code
), cur AS (SELECT * FROM sw WHERE period = 'current'),
   base AS (SELECT * FROM sw WHERE period = 'base'),
joined AS (
    SELECT
        coalesce(c.comparison, b.comparison) AS comparison,
        coalesce(c.store_code, b.store_code) AS store_code,
        coalesce(c.weeks_traded, 0) AS current_weeks,
        coalesce(b.weeks_traded, 0) AS base_weeks,
        coalesce(c.revenue, 0) AS current_revenue,
        coalesce(b.revenue, 0) AS base_revenue,
        coalesce(c.baskets, 0) AS current_baskets,
        coalesce(b.baskets, 0) AS base_baskets,
        coalesce(c.units, 0) AS current_units,
        coalesce(b.units, 0) AS base_units,
        coalesce(c.loyalty_baskets, 0) AS current_loyalty_baskets,
        coalesce(b.loyalty_baskets, 0) AS base_loyalty_baskets
    FROM cur c FULL OUTER JOIN base b
        ON c.comparison = b.comparison AND c.store_code = b.store_code
), metrics AS (
    SELECT
        j.comparison,
        j.store_code,
        st.store_format,
        st.store_region,
        st.lifecycle,
        j.current_weeks,
        j.base_weeks,
        (j.current_weeks = k.current_to - k.current_from + 1
         AND j.base_weeks = k.base_to - k.base_from + 1) AS is_lfl,
        j.current_revenue,
        j.base_revenue,
        j.current_revenue / nullif(j.base_revenue, 0) - 1 AS revenue_growth,
        j.current_revenue / nullif(j.current_weeks, 0) AS current_revenue_per_week,
        j.base_revenue / nullif(j.base_weeks, 0) AS base_revenue_per_week,
        j.current_baskets,
        j.base_baskets,
        j.current_baskets * 1.0 / nullif(j.base_baskets, 0) - 1 AS basket_growth,
        j.current_revenue / nullif(j.current_baskets, 0) AS current_avg_basket_value,
        j.base_revenue / nullif(j.base_baskets, 0) AS base_avg_basket_value,
        j.current_units * 1.0 / nullif(j.current_baskets, 0) AS current_units_per_basket,
        j.current_revenue / nullif(j.current_units, 0) AS current_avg_unit_price,
        j.current_loyalty_baskets * 1.0 / nullif(j.current_baskets, 0) AS current_loyalty_share
    FROM joined j
    JOIN dim_store st USING (store_code)
    JOIN dim_comparison k USING (comparison)
)
SELECT
    m.*,
    100 * m.current_revenue_per_week
        / nullif(median(m.current_revenue_per_week) OVER (PARTITION BY comparison, store_format), 0)
        AS revenue_index_vs_format,
    100 * m.current_avg_basket_value
        / nullif(median(m.current_avg_basket_value) OVER (PARTITION BY comparison, store_format), 0)
        AS basket_value_index_vs_format,
    CASE WHEN m.current_revenue_per_week IS NOT NULL THEN
        rank() OVER (
            PARTITION BY comparison, store_region
            ORDER BY m.current_revenue_per_week DESC NULLS LAST
        )
    END AS revenue_rank_in_region,
    CAST(count(m.current_revenue_per_week) OVER (PARTITION BY comparison, store_region) AS INTEGER)
        AS trading_stores_in_region,
    CASE WHEN m.is_lfl THEN
        (m.revenue_growth
         - median(CASE WHEN m.is_lfl THEN m.revenue_growth END)
               OVER (PARTITION BY comparison, store_format)) * 100
    END AS lfl_growth_gap_pp,
    CASE WHEN m.is_lfl THEN
        (m.revenue_growth - avg(CASE WHEN m.is_lfl THEN m.revenue_growth END)
               OVER (PARTITION BY comparison, store_format))
        / nullif(stddev_samp(CASE WHEN m.is_lfl THEN m.revenue_growth END)
               OVER (PARTITION BY comparison, store_format), 0)
    END AS lfl_growth_zscore
FROM metrics m
