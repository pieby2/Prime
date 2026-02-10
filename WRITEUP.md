# Write-Up: Sentiment vs Trader Performance on Hyperliquid

## Approach

I started by loading both datasets and checking for quality issues - both were clean (no missing values or duplicates). The fear/greed data had 2,644 daily readings while the trader data had 211K+ rows across 32 accounts.

For alignment, I merged on date using an inner join, which kept 211,218 rows spanning May 2023 to May 2025. I simplified the sentiment labels by grouping "Extreme Fear" with "Fear" and "Extreme Greed" with "Greed" to get three clean buckets.

From there I built daily per-account metrics: PnL (sum of closed trades), win rate (% profitable trades), trade count, average size, and long/short ratio. I also computed a leverage proxy using `Size USD / |Start Position|` since the dataset doesn't have an explicit leverage field. For statistical rigor, I used Mann-Whitney U tests to validate sentiment-based differences.

## What I Found

**1. Fear days are actually more profitable (p = 0.009)**

This was the most counterintuitive finding. Median daily PnL on Fear days ($826) beats Greed ($540) by a solid margin, and it's statistically significant. Looking deeper, traders with a strong long bias (>60% long ratio) earn $884 median on Fear vs $747 on Greed - suggesting they're catching mean-reversion bounces that others miss.

**2. Traders overtrade during fear but with smaller bets**

On Fear days, traders average 69 trades/day vs 46 on Greed, but median trade size drops ($1,802 vs $2,048). It looks like fear triggers more cautious, frequent activity - which isn't necessarily bad, since the win rate is also higher on Fear days (86.7% vs 84.1%, p = 0.0001).

**3. Regime transitions create outsized PnL dispersion**

Days when sentiment category shifts (e.g., Neutral to Fear) show roughly 2x the cross-trader PnL standard deviation compared to stable days ($19,927 vs $10,432). These transitional periods seem to separate skilled traders from the rest.

**4. Size matters more than frequency**

When I segmented traders, the high-size group averaged $326K total PnL vs $130K for small traders. Frequent traders do win more often (85.5% vs 83%) but the per-trade gains are comparable. The real differentiator is bet sizing, not trade count.

## Strategy Recommendations

**Strategy 1 - Contrarian long on fear (with reduced size)**

During Fear days, maintain a long bias but cut position size to 50% of normal. This captures the mean-reversion upside (Fear longs earn 18% more than Greed longs) while managing the deeper drawdowns that come with fear periods (mean drawdown -$22K on Fear vs -$11K on Neutral). Only for traders with >50% historical win rate.

**Strategy 2 - Cap frequency during greed**

When F&G goes above 65, high-size traders should limit daily trades to their Fear-day average. The data shows over-trading during Greed dilutes returns - you're chasing momentum that's already priced in. Keep the sizing, reduce the frequency.

## Limitations

- Only 32 accounts in the dataset, so segmentation results are directional rather than definitive
- No explicit leverage data - my proxy is an approximation
- Likely survivorship bias since the dataset probably over-represents active/successful traders
- The predictive model (71% accuracy) is a proof-of-concept, not production-ready
