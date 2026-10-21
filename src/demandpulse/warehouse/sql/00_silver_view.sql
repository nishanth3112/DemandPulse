-- Silver as a single view across the selected sources and all weeks.
-- hive_partitioning is off: shop_week is already a column inside each file.
CREATE OR REPLACE VIEW silver AS
SELECT
    shop_week,
    shop_date,
    shop_weekday,
    shop_hour,
    quantity,
    spend,
    prod_code,
    prod_code_10,
    prod_code_20,
    prod_code_30,
    prod_code_40,
    cust_code,
    seg_1,
    seg_2,
    basket_id,
    basket_size,
    basket_price_sensitivity,
    basket_type,
    basket_dominant_mission,
    store_code,
    store_format,
    store_region,
    is_loyalty,
    week_start,
    week_end
FROM read_parquet('$silver_pattern', hive_partitioning = false)
