# Business case study: where is revenue slipping, and could we see it coming?

A worked example of what the platform answers, with every figure labelled by how much weight it
can bear. The data is the synthetic dunnhumby dataset (761 stores, 117 weeks, revenue = `spend`).
There are no prices, costs, margins or promotions in it, so nothing below is about profit or about
the effect of any action.

| Label | Meaning |
|---|---|
| **Observed** | Measured directly from the data, and reconciled to the source |
| **Backtest** | How well a method would have predicted something that has since happened, using only data known at the time |
| **Scenario** | Arithmetic on a forecast: "if this driver moved by x, the gap would be y" |
| **Hypothetical** | A suggested way to use the results; untested, with no claimed benefit |

## The question

A chain's revenue looks flat or soft. Management wants to know whether it is slipping, where, and
why, and whether the stores that will miss next quarter's goal can be found before the quarter
starts.

## Observed: the last 13 weeks against the same weeks a year earlier

- **Revenue is slightly down.** 66.74M against 67.50M, -1.1%. Like-for-like, counting only the 756
  stores trading in both windows, it is -1.2%. Baskets are down 0.2% and the average basket is
  12.52, down 0.9%.
- **The decline comes from smaller baskets, not fewer visits.** Of the 0.75M net decline, units per
  basket explain -1.26M and baskets per week -0.19M, partly offset by higher revenue per unit
  (+0.53M), more trading weeks (+0.12M) and new stores (+0.04M).
- **Over the last 52 weeks the picture is mixed.** Total revenue is +0.6% on the prior 52 weeks but
  -0.3% like-for-like (743 comparable stores). Traffic (+2.31M) and basket size (+1.80M) rose and
  revenue per unit fell (-3.76M), the opposite of the last quarter. Price and mix cannot be
  separated: there are no list prices.
- **No region explains it.** All twelve regions are below last year like-for-like, from -0.05%
  (E01) to -2.07% (E03).
- **The average hides a wide spread across stores.** The median like-for-like store is -1.6%;
  154 of the 756 (20%) fell by more than 5% and 60 (8%) grew by more than 5%.
- **Categories pull in both directions.** The biggest drag is category G00004, 14.4% of revenue and
  -5.2%, taking 0.78 points off growth, followed by G00027 (-15.6%, -0.55 points). G00028 (+13.1%,
  +0.54 points) and G00029 (+39.1%, +0.44 points) add the most. The data has no product names, so
  categories are hierarchy codes.
- **Customers.** Loyalty-card baskets are 79.5% of baskets and 80.4% of revenue, so customer
  statistics describe those. 24.7% of customers (Champions) bring 60.7% of revenue, and 28.8%
  (Hibernating) bring 3.3%.

Source: the Gold marts, exported to the dashboard's Executive overview, Stores and regions and
Customers pages.

## Backtest: could the slip have been seen coming?

- **Forecasts.** An equal-weight average of three forecasters (seasonal naive, a 13-week trailing mean and LightGBM) had a 17.8% error per
  store-category series, 9.8% per store and 5.8% for the whole network over a 13-week horizon,
  against 21.7%, 10.6% and 3.7% for last year's value. It helps at store and series level but not
  for the network total. See [model-evaluation.md](model-evaluation.md).
- **Finding stores that miss a goal.** Using a hypothetical target (last year's revenue, unchanged)
  and the forecasts, ranking stores by their forecast gap relative to target put 96.1% real misses in the top 10% of stores,
  against 85.1% for ranking by each store's own recent growth and 65.0% for random choice
  (65.0% of stores missed in the test windows, so the lift is 1.48). The shortfall probabilities
  order stores well but are not calibrated chances. See [revenue-risk.md](revenue-risk.md).
- **Where it fails.** The top-ranked store at the last origin had been closed for about ten weeks,
  looked like a collapse to the forecast, and then beat its target. Large gaps are prompts to look,
  not findings.

## Scenario: how big a change would close the gap?

For the 20 riskiest stores at the last origin, the median uplift needed to reach the hypothetical
target is 13.9% (flat target) or 13.0% (growth-adjusted target). For the flat target, applying +2%
to traffic closes on average 12.5% of the gap, and +2% traffic, +2% basket size and +1% price/mix
together close 31.8%. No store reaches target in any preset scenario. This is arithmetic on the forecast: it says how large
a driver change would be needed, not that any action can produce it or what it would cost.

## Hypothetical: how the results could be used

One possible routine, untested here: each week, take the 20 stores with the most expected revenue
below target, read the evidence line for each (forecast gap, probability, own recent growth, weakest
driver), and have someone who knows the stores decide whether to look closer. The false alarm above
is the reason a person stays in the loop. No benefit is claimed. Whether this finds problems earlier
than existing reporting, and what acting on it is worth, can only be tested on real operations.

## What it would take to make this real

- **Real targets or budgets**, instead of last year's revenue.
- **Prices, costs and promotions**, to separate price from mix and to talk about margin.
- **Operational context** such as closures, refits and openings, so a forecast can be told that a
  dip was temporary.
- **Newer data** to validate the forecasts and the risk ranking on windows nobody has looked at, since
  the ensemble was chosen after seeing results on the same windows.
- **Monitoring** of forecast bias, since the last test window showed forecasts below target for 92%
  of stores when 65% really missed.
