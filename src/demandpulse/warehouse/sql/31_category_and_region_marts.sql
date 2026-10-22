-- Category performance per comparison: growth, share of revenue, penetration (share of all
-- baskets containing the category) and average unit price, current window vs base window.
-- A full outer join keeps categories that exist in only one window.
CREATE OR REPLACE TABLE mart_category_benchmark AS
WITH cat AS (
    SELECT w.comparison, w.period, f.prod_code_30,
           sum(f.revenue) AS revenue, sum(f.units) AS units, sum(f.baskets) AS cat_baskets
    FROM fct_store_category_week f JOIN dim_comparison_week w USING (shop_week)
    GROUP BY ALL
), tot AS (
    SELECT w.comparison, w.period, sum(s.revenue) AS revenue, sum(s.baskets) AS baskets
    FROM fct_store_week s JOIN dim_comparison_week w USING (shop_week)
    GROUP BY ALL
), enriched AS (
    SELECT c.comparison, c.period, c.prod_code_30,
           CAST(c.revenue AS DOUBLE) AS revenue,
           CAST(c.units AS DOUBLE) AS units,
           CAST(c.revenue AS DOUBLE) / t.revenue AS share,
           CAST(c.cat_baskets AS DOUBLE) / t.baskets AS penetration,
           CAST(c.revenue AS DOUBLE) / nullif(c.units, 0) AS avg_unit_price
    FROM cat c JOIN tot t USING (comparison, period)
), cur AS (SELECT * FROM enriched WHERE period = 'current'),
   base AS (SELECT * FROM enriched WHERE period = 'base')
SELECT
    coalesce(cur.comparison, base.comparison) AS comparison,
    coalesce(cur.prod_code_30, base.prod_code_30) AS prod_code_30,
    coalesce(cur.revenue, 0) AS current_revenue,
    coalesce(base.revenue, 0) AS base_revenue,
    cur.revenue / nullif(base.revenue, 0) - 1 AS revenue_growth,
    coalesce(cur.share, 0) AS current_share,
    coalesce(base.share, 0) AS base_share,
    (coalesce(cur.share, 0) - coalesce(base.share, 0)) * 100 AS share_change_pp,
    -- Percentage points of national revenue growth contributed by this category; the column
    -- sums to the national growth rate (x 100) within a comparison.
    (coalesce(cur.revenue, 0) - coalesce(base.revenue, 0))
        / sum(coalesce(base.revenue, 0)) OVER (PARTITION BY coalesce(cur.comparison, base.comparison))
        * 100 AS contribution_to_growth_pp,
    cur.penetration AS current_penetration,
    base.penetration AS base_penetration,
    cur.avg_unit_price AS current_avg_unit_price,
    base.avg_unit_price AS base_avg_unit_price,
    cur.avg_unit_price / nullif(base.avg_unit_price, 0) - 1 AS avg_unit_price_change
FROM cur FULL OUTER JOIN base
    ON cur.comparison = base.comparison AND cur.prod_code_30 = base.prod_code_30;

-- Region benchmark with like-for-like (LFL) growth. A store is LFL for a comparison when it
-- traded in every week of both windows, which removes openings and closures from the growth
-- figure. Region 'ALL' is the national total.
CREATE OR REPLACE TABLE mart_region_benchmark AS
WITH sw AS (
    SELECT w.comparison, w.period, s.store_code, min(s.store_region) AS store_region,
           count(*) AS weeks_traded, sum(s.revenue) AS revenue, sum(s.baskets) AS baskets
    FROM fct_store_week s JOIN dim_comparison_week w USING (shop_week)
    GROUP BY w.comparison, w.period, s.store_code
), cur AS (SELECT * FROM sw WHERE period = 'current'),
   base AS (SELECT * FROM sw WHERE period = 'base'),
joined AS (
    SELECT
        coalesce(c.comparison, b.comparison) AS comparison,
        coalesce(c.store_code, b.store_code) AS store_code,
        coalesce(c.store_region, b.store_region) AS store_region,
        coalesce(c.weeks_traded, 0) AS cur_weeks,
        coalesce(b.weeks_traded, 0) AS base_weeks,
        CAST(coalesce(c.revenue, 0) AS DOUBLE) AS cur_rev,
        CAST(coalesce(b.revenue, 0) AS DOUBLE) AS base_rev,
        coalesce(c.baskets, 0) AS cur_baskets,
        coalesce(b.baskets, 0) AS base_baskets
    FROM cur c FULL OUTER JOIN base b
        ON c.comparison = b.comparison AND c.store_code = b.store_code
), flagged AS (
    SELECT j.*,
           (j.cur_weeks = k.current_to - k.current_from + 1
            AND j.base_weeks = k.base_to - k.base_from + 1) AS is_lfl
    FROM joined j JOIN dim_comparison k USING (comparison)
)
SELECT
    comparison,
    coalesce(store_region, 'ALL') AS store_region,
    CAST(count(*) FILTER (WHERE cur_weeks > 0) AS INTEGER) AS stores_current,
    CAST(count(*) FILTER (WHERE base_weeks > 0) AS INTEGER) AS stores_base,
    CAST(count(*) FILTER (WHERE is_lfl) AS INTEGER) AS lfl_stores,
    sum(cur_rev) AS current_revenue,
    sum(base_rev) AS base_revenue,
    sum(cur_rev) / nullif(sum(base_rev), 0) - 1 AS revenue_growth,
    sum(cur_rev) FILTER (WHERE is_lfl) AS lfl_current_revenue,
    sum(base_rev) FILTER (WHERE is_lfl) AS lfl_base_revenue,
    sum(cur_rev) FILTER (WHERE is_lfl) / nullif(sum(base_rev) FILTER (WHERE is_lfl), 0) - 1
        AS lfl_revenue_growth,
    sum(cur_baskets) AS current_baskets,
    sum(base_baskets) AS base_baskets,
    sum(cur_baskets) * 1.0 / nullif(sum(base_baskets), 0) - 1 AS basket_growth,
    sum(cur_rev) / nullif(sum(cur_baskets), 0) AS current_avg_basket_value,
    sum(base_rev) / nullif(sum(base_baskets), 0) AS base_avg_basket_value
FROM flagged
GROUP BY GROUPING SETS ((comparison, store_region), (comparison));

-- Regional mix index for the current window: a region's category share relative to the
-- national share (100 = in line with national, 120 = over-indexes by 20%).
CREATE OR REPLACE TABLE mart_region_category_mix AS
WITH rc AS (
    SELECT w.comparison, f.store_region, f.prod_code_30, sum(f.revenue) AS revenue
    FROM fct_store_category_week f JOIN dim_comparison_week w USING (shop_week)
    WHERE w.period = 'current'
    GROUP BY ALL
), region_total AS (
    SELECT comparison, store_region, sum(revenue) AS revenue FROM rc GROUP BY ALL
), national AS (
    SELECT comparison, prod_code_30, sum(revenue) AS revenue FROM rc GROUP BY ALL
), national_total AS (
    SELECT comparison, sum(revenue) AS revenue FROM rc GROUP BY ALL
)
SELECT
    rc.comparison,
    rc.store_region,
    rc.prod_code_30,
    CAST(rc.revenue AS DOUBLE) AS revenue,
    CAST(rc.revenue AS DOUBLE) / rt.revenue AS region_share,
    CAST(n.revenue AS DOUBLE) / nt.revenue AS national_share,
    100 * (CAST(rc.revenue AS DOUBLE) / rt.revenue) / (CAST(n.revenue AS DOUBLE) / nt.revenue)
        AS mix_index
FROM rc
JOIN region_total rt USING (comparison, store_region)
JOIN national n USING (comparison, prod_code_30)
JOIN national_total nt USING (comparison)
