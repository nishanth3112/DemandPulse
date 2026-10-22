-- Basket fact: one row per basket_id. This is the single heavy pass over Silver; store-week
-- KPIs, customer marts and dimensions are derived from it instead of rescanning 300M lines.
-- Basket attributes should be constant within a basket, which makes min() exact. `consistent`
-- proves it per basket by comparing min with max of every attribute in the same pass, and the
-- Gold validation fails if any basket is not consistent.
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
    (min(shop_week) = max(shop_week)
        AND min(shop_date) = max(shop_date)
        AND min(shop_weekday) = max(shop_weekday)
        AND min(shop_hour) = max(shop_hour)
        AND min(store_code) = max(store_code)
        AND min(store_format) = max(store_format)
        AND min(store_region) = max(store_region)
        AND coalesce(min(cust_code) = max(cust_code), true)
        AND min(basket_size) = max(basket_size)
        AND min(basket_type) = max(basket_type)
        AND min(basket_dominant_mission) = max(basket_dominant_mission)
        AND min(basket_price_sensitivity) = max(basket_price_sensitivity)
    ) AS consistent,
    CAST(count(*) AS INTEGER) AS lines,
    CAST(sum(quantity) AS BIGINT) AS units,
    CAST(sum(spend) AS DECIMAL(18, 2)) AS revenue
FROM silver
GROUP BY basket_id
