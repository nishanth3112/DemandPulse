INSERT INTO fct_store_category_week
SELECT
    shop_week,
    store_code,
    prod_code_30,
    min(store_format),
    min(store_region),
    CAST(sum(spend) AS DECIMAL(18, 2)),
    CAST(sum(quantity) AS BIGINT),
    CAST(count(*) AS INTEGER),
    CAST(count(DISTINCT basket_id) AS INTEGER),
    CAST(sum(CASE WHEN is_loyalty THEN spend ELSE 0 END) AS DECIMAL(18, 2))
FROM read_parquet('$silver_week', hive_partitioning = false)
GROUP BY shop_week, store_code, prod_code_30
