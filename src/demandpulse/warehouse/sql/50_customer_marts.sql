-- RFM over the last $year_weeks weeks, loyalty customers only (no-card baskets have no
-- customer). Recency = weeks between a customer's last basket and the last data week
-- (0 = shopped in the final week). Frequency = baskets, monetary = revenue in the window.
-- Scores are 1-5 with 5 best, from ceil(5 * cume_dist) so tied customers always share a score.
CREATE OR REPLACE TABLE mart_customer_rfm AS
WITH bounds AS (SELECT max(week_idx) AS last_idx FROM dim_date),
activity AS (
    SELECT b.cust_code,
           count(*) AS baskets,
           CAST(sum(b.revenue) AS DOUBLE) AS revenue,
           max(d.week_idx) AS last_week_idx,
           min(d.week_idx) AS first_week_idx_in_window
    FROM fct_basket b
    JOIN dim_date d USING (shop_week)
    CROSS JOIN bounds
    WHERE b.cust_code IS NOT NULL AND d.week_idx > bounds.last_idx - $year_weeks
    GROUP BY b.cust_code
), scored AS (
    SELECT a.*,
           bounds.last_idx - a.last_week_idx AS recency_weeks,
           CAST(ceil(5.0 * cume_dist() OVER (ORDER BY bounds.last_idx - a.last_week_idx DESC)) AS INTEGER)
               AS r_score,
           CAST(ceil(5.0 * cume_dist() OVER (ORDER BY a.baskets)) AS INTEGER) AS f_score,
           CAST(ceil(5.0 * cume_dist() OVER (ORDER BY a.revenue)) AS INTEGER) AS m_score
    FROM activity a CROSS JOIN bounds
)
SELECT
    s.cust_code,
    c.seg_1,
    c.seg_2,
    c.first_week AS first_week_overall,
    s.baskets,
    s.revenue,
    s.recency_weeks,
    s.r_score,
    s.f_score,
    s.m_score,
    CASE
        WHEN s.r_score >= 4 AND s.f_score >= 4 AND s.m_score >= 4 THEN 'Champions'
        WHEN s.r_score >= 3 AND s.f_score >= 4 THEN 'Loyal'
        WHEN s.r_score = 5 AND s.f_score = 1 THEN 'New or returning'
        WHEN s.r_score >= 4 AND s.f_score >= 2 THEN 'Potential loyalists'
        WHEN s.r_score <= 2 AND s.f_score >= 4 THEN 'At risk'
        WHEN s.r_score <= 2 AND s.f_score <= 2 THEN 'Hibernating'
        ELSE 'Needs attention'
    END AS rfm_segment
FROM scored s JOIN dim_customer c USING (cust_code);

CREATE OR REPLACE TABLE mart_rfm_segment_summary AS
SELECT
    rfm_segment,
    CAST(count(*) AS INTEGER) AS customers,
    count(*) * 1.0 / sum(count(*)) OVER () AS customer_share,
    sum(revenue) AS revenue,
    sum(revenue) / sum(sum(revenue)) OVER () AS revenue_share,
    avg(baskets) AS avg_baskets,
    avg(revenue) AS avg_revenue,
    avg(recency_weeks) AS avg_recency_weeks
FROM mart_customer_rfm
GROUP BY rfm_segment;

-- Cohort retention by 4-week period. A customer's cohort is the first period they appear in
-- the data. Cohort 1 is the baseline: those customers were already shopping before the data
-- starts, so they are NOT acquisitions and their "first" period is left-censored. The final
-- period can be shorter than 4 weeks (weeks_in_period) which understates its activity.
CREATE OR REPLACE TABLE mart_customer_cohort AS
WITH periods AS (
    SELECT period_idx, min(week_start) AS period_start, CAST(count(*) AS INTEGER) AS weeks_in_period
    FROM dim_date GROUP BY period_idx
), cust_period AS (
    SELECT b.cust_code, d.period_idx, count(*) AS baskets, sum(b.revenue) AS revenue
    FROM fct_basket b JOIN dim_date d USING (shop_week)
    WHERE b.cust_code IS NOT NULL
    GROUP BY b.cust_code, d.period_idx
), first_period AS (
    SELECT cust_code, min(period_idx) AS cohort_period FROM cust_period GROUP BY cust_code
), cohort_size AS (
    SELECT cohort_period, count(*) AS cohort_customers FROM first_period GROUP BY cohort_period
)
SELECT
    f.cohort_period,
    cp.period_idx,
    cp.period_idx - f.cohort_period AS periods_since_first,
    cohort_start.period_start AS cohort_start,
    p.period_start,
    p.weeks_in_period,
    CAST(cs.cohort_customers AS INTEGER) AS cohort_customers,
    CAST(count(*) AS INTEGER) AS active_customers,
    count(*) * 1.0 / cs.cohort_customers AS retention_rate,
    CAST(sum(cp.baskets) AS BIGINT) AS baskets,
    CAST(sum(cp.revenue) AS DECIMAL(18, 2)) AS revenue,
    CAST(sum(cp.revenue) AS DOUBLE) / count(*) AS revenue_per_active_customer,
    (f.cohort_period = 1) AS is_baseline_cohort
FROM cust_period cp
JOIN first_period f USING (cust_code)
JOIN cohort_size cs ON cs.cohort_period = f.cohort_period
JOIN periods p ON p.period_idx = cp.period_idx
JOIN periods cohort_start ON cohort_start.period_idx = f.cohort_period
GROUP BY f.cohort_period, cp.period_idx, cohort_start.period_start, p.period_start,
         p.weeks_in_period, cs.cohort_customers
