-- KPI definitions (all observed measurements from the dataset):
--   revenue          sum of spend
--   units            sum of quantity
--   lines            product lines (rows)
--   baskets          distinct basket_id (traffic proxy: transactions at till, not footfall)
--   customers        distinct loyalty customers (no-card baskets have no customer)
--   avg_basket_value revenue / baskets
--   units_per_basket units / baskets
--   lines_per_basket lines / baskets
--   avg_unit_price   revenue / units, an average price paid across a mixed product set,
--                    not a list price and not comparable across different product mixes
--   loyalty_basket_share  share of baskets with a customer code
CREATE OR REPLACE TABLE fct_store_week AS
SELECT
    shop_week,
    store_code,
    min(store_format) AS store_format,
    min(store_region) AS store_region,
    CAST(count(*) AS INTEGER) AS baskets,
    CAST(sum(revenue) AS DECIMAL(18, 2)) AS revenue,
    CAST(sum(units) AS BIGINT) AS units,
    CAST(sum(lines) AS BIGINT) AS lines,
    CAST(count(DISTINCT cust_code) AS INTEGER) AS customers,
    CAST(sum(CASE WHEN is_loyalty THEN 1 ELSE 0 END) AS INTEGER) AS loyalty_baskets,
    CAST(sum(CASE WHEN is_loyalty THEN revenue ELSE 0 END) AS DECIMAL(18, 2)) AS loyalty_revenue,
    CAST(sum(revenue) AS DOUBLE) / count(*) AS avg_basket_value,
    CAST(sum(units) AS DOUBLE) / count(*) AS units_per_basket,
    CAST(sum(lines) AS DOUBLE) / count(*) AS lines_per_basket,
    CAST(sum(revenue) AS DOUBLE) / nullif(sum(units), 0) AS avg_unit_price,
    sum(CASE WHEN is_loyalty THEN 1 ELSE 0 END) * 1.0 / count(*) AS loyalty_basket_share
FROM fct_basket
GROUP BY shop_week, store_code;

CREATE OR REPLACE TABLE mart_basket_mix AS
SELECT
    shop_week,
    store_format,
    basket_type,
    basket_size,
    basket_dominant_mission,
    basket_price_sensitivity,
    is_loyalty,
    CAST(count(*) AS INTEGER) AS baskets,
    CAST(sum(revenue) AS DECIMAL(18, 2)) AS revenue,
    CAST(sum(units) AS BIGINT) AS units,
    CAST(sum(lines) AS BIGINT) AS lines
FROM fct_basket
GROUP BY ALL;

-- When shoppers trade: baskets and revenue by store format, weekday (1 = Sunday) and hour,
-- over the whole data period, with each cell's share of its format's revenue.
CREATE OR REPLACE TABLE mart_trading_pattern AS
SELECT
    store_format,
    shop_weekday,
    shop_hour,
    CAST(count(*) AS BIGINT) AS baskets,
    CAST(sum(revenue) AS DECIMAL(18, 2)) AS revenue,
    CAST(sum(revenue) AS DOUBLE)
        / sum(sum(revenue)) OVER (PARTITION BY store_format) AS revenue_share_of_format
FROM fct_basket
GROUP BY store_format, shop_weekday, shop_hour
