-- Basket fact: one row per basket_id. This is the single heavy pass over Silver; store-week
-- KPIs, customer marts and dimensions are derived from it instead of rescanning 300M lines.
-- Basket attributes are constant within a basket (validated separately), so min() is exact.
CREATE OR REPLACE TABLE fct_basket AS
SELECT
    basket_id,
    min(shop_week) AS shop_week,
    min(shop_date) AS shop_date,
    min(shop_weekday) AS shop_weekday,
    min(shop_hour) AS shop_hour,
    min(store_code) AS store_code,
    min(store_format) AS store_format,
    min(store_region) AS store_region,
    min(cust_code) AS cust_code,
    bool_or(is_loyalty) AS is_loyalty,
    min(seg_1) AS seg_1,
    min(seg_2) AS seg_2,
    min(basket_size) AS basket_size,
    min(basket_type) AS basket_type,
    min(basket_dominant_mission) AS basket_dominant_mission,
    min(basket_price_sensitivity) AS basket_price_sensitivity,
    CAST(count(*) AS INTEGER) AS lines,
    CAST(sum(quantity) AS BIGINT) AS units,
    CAST(sum(spend) AS DECIMAL(18, 2)) AS revenue
FROM silver
GROUP BY basket_id
