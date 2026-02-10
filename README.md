# Trader Performance vs Market Sentiment

Primetrade.ai Data Science Intern assignment.

## What this does

Analyzes how Bitcoin market sentiment (Fear & Greed Index) correlates with trader behavior and performance on Hyperliquid. Uses ~211K trades spanning May 2023 to May 2025 merged with daily sentiment data. The analysis identifies statistically significant patterns and proposes two trading strategies backed by the data.

## Project structure

```
analysis.py              - main script, runs the full analysis + generates charts
dashboard.py             - interactive Streamlit dashboard
fear_greed_index.csv     - Bitcoin Fear/Greed Index data
historical_data.csv      - Hyperliquid trader data
output_charts/           - generated after running analysis.py
requirements.txt         - python dependencies
WRITEUP.md               - methodology, insights, and strategy writeup
```

## How to run

You need Python 3.8+ installed.

```bash
pip install -r requirements.txt
python analysis.py
```

This prints summary tables and statistical tests to the console, and saves all charts to `./output_charts/`.

To launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

## Data overview

The Fear/Greed dataset has 2,644 daily readings from Feb 2018 onwards. The trader dataset has 211,224 rows covering 32 accounts trading 246 different coins. After merging on date (inner join), we end up with 211,218 rows.

## Main findings

1. **Fear days are actually more profitable** - median daily PnL on Fear days ($826) is noticeably higher than Greed days ($540). Mann-Whitney U test confirms this is significant (p = 0.009).

2. **Traders are more active during fear but bet smaller** - average 69 trades/day on Fear vs 46 on Greed, but the median trade size flips ($1,802 vs $2,048). Seems like fear drives more cautious, frequent trading.

3. **Sentiment regime changes create volatility spikes** - when sentiment flips between categories, the cross-trader PnL standard deviation roughly doubles (~$19.9K vs ~$10.4K on stable days).

## Strategy ideas

- **On Fear days**: go long but with half your usual position size. The mean-reversion works, but drawdowns are also deeper so you want to control risk.
- **On Greed days (F&G > 65)**: if you're a high-size trader, cap your trade count at your Fear-day average. Over-trading during greed periods dilutes PnL.

## Bonus stuff

- Gradient Boosting model predicting next-day profitability buckets (71% CV accuracy)
- K-Means clustering identifying 4 trader archetypes
- Streamlit dashboard with interactive filters and Plotly charts
