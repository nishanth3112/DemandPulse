-- Store x category x week. "category" is product hierarchy level 30 (31 groups on the data).
-- baskets = distinct baskets containing the category in that store-week, so summing over
-- store-weeks gives exact distinct basket counts (a basket belongs to one store-week).
-- Built one week at a time (see 30_store_category_week_weekly.sql): counting distinct baskets
-- across all 117 weeks in one pass exceeded the memory limit on the full dataset.
CREATE OR REPLACE TABLE fct_store_category_week (
    shop_week INTEGER,
    store_code VARCHAR,
    prod_code_30 VARCHAR,
    store_format VARCHAR,
    store_region VARCHAR,
    revenue DECIMAL(18, 2),
    units BIGINT,
    lines INTEGER,
    baskets INTEGER,
    loyalty_revenue DECIMAL(18, 2)
)
