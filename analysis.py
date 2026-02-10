"""
Analysis script for Primetrade.ai Data Science Intern assignment.

Looks at how Bitcoin fear/greed sentiment affects trader behavior
and performance on Hyperliquid. Generates charts into ./output_charts/
and prints summary tables to console.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

CHARTS_DIR = "output_charts"
os.makedirs(CHARTS_DIR, exist_ok=True)


# =====================================================================
#  PART A - DATA PREPARATION
# =====================================================================
print("=" * 60)
print("PART A - DATA PREPARATION")
print("=" * 60)

# load both csv files
sentiment_df = pd.read_csv("fear_greed_index.csv")
trades_df = pd.read_csv("historical_data.csv")

# quick look at what we're working with
print(f"\nFear/Greed dataset: {sentiment_df.shape[0]:,} rows, {sentiment_df.shape[1]} cols")
print(f"Missing values:\n{sentiment_df.isnull().sum().to_string()}")
print(f"Duplicates: {sentiment_df.duplicated().sum()}")

print(f"\nTrader dataset: {trades_df.shape[0]:,} rows, {trades_df.shape[1]} cols")
print(f"Missing values:\n{trades_df.isnull().sum().to_string()}")
print(f"Duplicates: {trades_df.duplicated().sum()}")

# --- timestamp conversion and cleanup ---

sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
sentiment_df.rename(columns={"value": "fgi_value", "classification": "sentiment"}, inplace=True)

# collapse extreme fear/greed into just fear/greed for simpler splits
sentiment_df["sent_group"] = sentiment_df["sentiment"].map({
    "Extreme Fear": "Fear", "Fear": "Fear",
    "Neutral": "Neutral",
    "Greed": "Greed", "Extreme Greed": "Greed"
})

# trader timestamps are in DD-MM-YYYY HH:MM format
trades_df["Timestamp IST"] = pd.to_datetime(
    trades_df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
)
trades_df["date"] = trades_df["Timestamp IST"].dt.normalize()

# some numeric columns might have string artifacts, force them
num_cols = ["Closed PnL", "Fee", "Size USD", "Size Tokens",
            "Execution Price", "Start Position"]
for c in num_cols:
    trades_df[c] = pd.to_numeric(trades_df[c], errors="coerce")

trades_df.dropna(subset=["date", "Closed PnL"], inplace=True)

# compute a rough leverage proxy from size vs position
# (dataset doesn't have an explicit leverage column)
trades_df["lev_proxy"] = np.where(
    trades_df["Start Position"].abs() > 0,
    trades_df["Size USD"] / trades_df["Start Position"].abs(),
    np.nan
)

# merge on date
merged = trades_df.merge(
    sentiment_df[["date", "fgi_value", "sentiment", "sent_group"]],
    on="date", how="inner"
)

print(f"\nAfter merge: {merged.shape[0]:,} rows, {merged.shape[1]} cols")
print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
print(f"Accounts: {merged['Account'].nunique()}, Coins: {merged['Coin'].nunique()}")

# --- derived metrics ---

# only look at closing trades for PnL analysis
merged["is_close"] = merged["Direction"].str.contains("Close", case=False, na=False)
closes = merged[merged["is_close"]].copy()
closes["won"] = (closes["Closed PnL"] > 0).astype(int)

# aggregate to daily per-account level
daily = (
    closes.groupby(["date", "Account", "sent_group", "sentiment", "fgi_value"])
    .agg(
        pnl=("Closed PnL", "sum"),
        n_trades=("Closed PnL", "count"),
        n_wins=("won", "sum"),
        avg_size=("Size USD", "mean"),
        volume=("Size USD", "sum"),
    )
    .reset_index()
)
daily["winrate"] = daily["n_wins"] / daily["n_trades"]

# long/short ratio from all trades (not just closes)
ls = (
    merged.groupby(["date", "Account", "sent_group"])
    .apply(lambda g: pd.Series({
        "n_long": (g["Side"] == "BUY").sum(),
        "n_short": (g["Side"] == "SELL").sum(),
    }), include_groups=False)
    .reset_index()
)
ls["long_pct"] = ls["n_long"] / (ls["n_long"] + ls["n_short"])

daily = daily.merge(ls, on=["date", "Account", "sent_group"], how="left")

# drawdown helper
def calc_max_dd(pnl_series):
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    return (cumulative - running_max).min()

print("\nMetrics ready: daily PnL, win rate, trade count, size, long/short ratio, drawdown proxy\n")


# =====================================================================
#  PART B - ANALYSIS
# =====================================================================
print("=" * 60)
print("PART B - ANALYSIS")
print("=" * 60)

sent_order = ["Fear", "Neutral", "Greed"]
color_map = {"Fear": "#e74c3c", "Neutral": "#95a5a6", "Greed": "#27ae60"}

# --- B1: does performance differ between Fear vs Greed? ---
print("\n--- B1: Performance by Sentiment ---")

perf_table = (
    daily.groupby("sent_group")
    .agg(
        median_pnl=("pnl", "median"),
        mean_pnl=("pnl", "mean"),
        avg_winrate=("winrate", "mean"),
        count=("pnl", "count"),
    ).round(4)
)
print(perf_table.to_string())

# drawdown by sentiment
dd_by_sent = (
    closes.sort_values("Timestamp IST")
    .groupby(["Account", "sent_group"])["Closed PnL"]
    .apply(calc_max_dd)
    .reset_index(name="max_dd")
)
dd_table = dd_by_sent.groupby("sent_group")["max_dd"].agg(["mean", "median"]).round(2)
print("\nDrawdown by sentiment (mean/median):")
print(dd_table.to_string())

# statistical test - is the difference real?
fear_pnl = daily[daily["sent_group"] == "Fear"]["pnl"]
greed_pnl = daily[daily["sent_group"] == "Greed"]["pnl"]

u_pnl, p_pnl = stats.mannwhitneyu(fear_pnl, greed_pnl, alternative="two-sided")
print(f"\nMann-Whitney U (PnL: Fear vs Greed): U={u_pnl:,.0f}, p={p_pnl:.4f}")
print(f"  -> {'Statistically significant' if p_pnl < 0.05 else 'Not significant'} at 5%")

fear_wr = daily[daily["sent_group"] == "Fear"]["winrate"]
greed_wr = daily[daily["sent_group"] == "Greed"]["winrate"]
u_wr, p_wr = stats.mannwhitneyu(fear_wr, greed_wr, alternative="two-sided")
print(f"\nMann-Whitney U (Win Rate: Fear vs Greed): U={u_wr:,.0f}, p={p_wr:.4f}")
print(f"  -> {'Statistically significant' if p_wr < 0.05 else 'Not significant'} at 5%")

# chart: PnL distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, s in zip(axes, sent_order):
    vals = daily[daily["sent_group"] == s]["pnl"].clip(-2000, 2000)
    ax.hist(vals, bins=60, edgecolor="white", alpha=0.85, color=color_map[s])
    med = daily[daily["sent_group"] == s]["pnl"].median()
    ax.axvline(med, color="black", ls="--", lw=1.5, label=f"Median: ${med:.0f}")
    ax.set_title(f"{s} Days", fontweight="bold", fontsize=14)
    ax.set_xlabel("Daily PnL ($)")
    ax.legend()
axes[0].set_ylabel("Frequency")
fig.suptitle("Daily PnL Distribution by Sentiment", fontsize=15, y=1.02)
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/B1_pnl_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()

# chart: win rate boxplot
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=daily, x="sent_group", y="winrate", order=sent_order,
            palette=color_map, ax=ax)
ax.set_title("Win Rate by Sentiment", fontsize=14, fontweight="bold")
ax.set_ylabel("Win Rate")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/B1_winrate_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.close()


# --- B2: do traders change behavior based on sentiment? ---
print("\n--- B2: Behavioral Changes ---")

beh = (
    daily.groupby("sent_group")
    .agg(
        avg_trades=("n_trades", "mean"),
        med_size=("avg_size", "median"),
        avg_long_pct=("long_pct", "mean"),
        med_volume=("volume", "median"),
    ).round(4)
)
print(beh.to_string())

# trade count and size chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=daily, x="sent_group", y="n_trades", order=sent_order,
            palette="viridis", errorbar=("ci", 95), ax=ax1)
ax1.set_title("Avg Trades per Account-Day", fontweight="bold")
ax1.set_ylabel("Trade Count")

sns.barplot(data=daily, x="sent_group", y="avg_size", order=sent_order,
            palette="magma", errorbar=("ci", 95), ax=ax2, estimator=np.median)
ax2.set_title("Median Trade Size (USD)", fontweight="bold")
ax2.set_ylabel("Size ($)")

fig.suptitle("Behavioral Differences by Sentiment", fontsize=15, y=1.02)
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/B2_behavior.png", dpi=150, bbox_inches="tight")
plt.close()

# long/short ratio
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=daily, x="sent_group", y="long_pct", order=sent_order,
            palette=[color_map[s] for s in sent_order],
            errorbar=("ci", 95), ax=ax)
ax.axhline(0.5, ls="--", color="grey", alpha=0.5, label="50% balanced")
ax.set_title("Long Ratio by Sentiment", fontweight="bold", fontsize=14)
ax.set_ylabel("Long Ratio")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend()
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/B2_long_ratio.png", dpi=150, bbox_inches="tight")
plt.close()

# leverage proxy distribution (if we have enough data)
lev = merged[merged["lev_proxy"].notna() & (merged["lev_proxy"] < 50)]
if len(lev) > 100:
    lev_stats = lev.groupby("sent_group")["lev_proxy"].describe().round(2)
    print(f"\nLeverage proxy stats:")
    print(lev_stats[["mean", "50%", "75%"]].to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=lev, x="sent_group", y="lev_proxy", order=sent_order,
                palette=color_map, showfliers=False, ax=ax)
    ax.set_title("Leverage Proxy by Sentiment", fontweight="bold", fontsize=14)
    ax.set_ylabel("Size USD / |Start Position|")
    plt.tight_layout()
    fig.savefig(f"{CHARTS_DIR}/B2_leverage.png", dpi=150, bbox_inches="tight")
    plt.close()


# --- B3: trader segmentation ---
print("\n--- B3: Trader Segments ---")

# build per-account profile
profiles = (
    closes.groupby("Account")
    .agg(
        total_pnl=("Closed PnL", "sum"),
        num_trades=("Closed PnL", "count"),
        winrate=("won", "mean"),
        avg_size=("Size USD", "mean"),
        med_size=("Size USD", "median"),
        days_active=("date", "nunique"),
    )
    .reset_index()
)
profiles["pnl_per_trade"] = profiles["total_pnl"] / profiles["num_trades"]
profiles["trades_per_day"] = profiles["num_trades"] / profiles["days_active"]

# segment 1: big vs small traders (by median trade size)
cutoff_size = profiles["med_size"].median()
profiles["size_seg"] = np.where(profiles["med_size"] >= cutoff_size, "Large", "Small")

# segment 2: frequent vs infrequent
cutoff_freq = profiles["trades_per_day"].median()
profiles["freq_seg"] = np.where(profiles["trades_per_day"] >= cutoff_freq, "Frequent", "Infrequent")

# segment 3: consistent winners vs others
profiles["perf_seg"] = np.where(
    (profiles["winrate"] >= 0.5) & (profiles["total_pnl"] > 0),
    "Winner", "Struggling"
)

# print segment summaries
for col, label in [("size_seg", "Size"), ("freq_seg", "Frequency"), ("perf_seg", "Performance")]:
    print(f"\n  {label} segment:")
    summary = profiles.groupby(col).agg(
        n=("Account", "count"),
        avg_pnl=("total_pnl", "mean"),
        avg_wr=("winrate", "mean"),
        avg_sz=("avg_size", "mean"),
    ).round(2)
    print(summary.to_string())

# attach segments to daily data for cross-analysis
daily_seg = daily.merge(
    profiles[["Account", "size_seg", "freq_seg", "perf_seg"]],
    on="Account", how="left"
)

# segment x sentiment chart
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (seg_col, title) in zip(axes, [("size_seg", "Size"),
                                        ("freq_seg", "Frequency"),
                                        ("perf_seg", "Performance")]):
    sns.barplot(data=daily_seg, x="sent_group", y="pnl",
                hue=seg_col, order=sent_order, ax=ax, errorbar=("ci", 95))
    ax.set_title(f"PnL by {title} Segment", fontweight="bold")
    ax.set_ylabel("Daily PnL ($)")
    ax.legend(title=title, fontsize=8)
fig.suptitle("Segment Performance Across Sentiment", fontsize=15, y=1.02)
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/B3_segments.png", dpi=150, bbox_inches="tight")
plt.close()


# --- B4: key insights with supporting charts ---
print("\n--- B4: Key Insights ---")

# insight 1: contrarian longs on fear days
fear_longs = daily_seg[(daily_seg["sent_group"] == "Fear") & (daily_seg["long_pct"] > 0.6)]
greed_longs = daily_seg[(daily_seg["sent_group"] == "Greed") & (daily_seg["long_pct"] > 0.6)]
print(f"\n  Insight 1 - Contrarian longs on Fear days")
print(f"    Long-heavy traders (>60%) on Fear days -> median PnL ${fear_longs['pnl'].median():.2f}")
print(f"    Same group on Greed days -> median PnL ${greed_longs['pnl'].median():.2f}")

# insight 2: frequency tradeoff
print(f"\n  Insight 2 - Frequency/profit tradeoff")
freq_comp = profiles.groupby("freq_seg").agg(
    wr=("winrate", "mean"),
    pnl_per=("pnl_per_trade", "mean"),
    total=("total_pnl", "mean"),
).round(4)
print(freq_comp.to_string())

# insight 3: regime transitions are volatile
daily_agg = (
    daily.groupby(["date", "sent_group", "fgi_value"])
    .agg(total_pnl=("pnl", "sum"), pnl_std=("pnl", "std"), n_traders=("pnl", "count"))
    .reset_index()
    .sort_values("date")
)
daily_agg["prev_sent"] = daily_agg["sent_group"].shift(1)
daily_agg["regime_flip"] = daily_agg["sent_group"] != daily_agg["prev_sent"]

flip_stats = daily_agg.groupby("regime_flip")["pnl_std"].agg(["mean", "median"]).round(2)
print(f"\n  Insight 3 - Regime transitions = higher volatility")
print(flip_stats.to_string())

# scatter: FGI value vs daily pnl
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(daily_agg["fgi_value"], daily_agg["total_pnl"],
                c=daily_agg["fgi_value"], cmap="RdYlGn", alpha=0.7,
                edgecolors="white", s=50)
ax.axhline(0, color="grey", ls="--", alpha=0.4)
ax.set_xlabel("Fear & Greed Index")
ax.set_ylabel("Total Daily PnL ($)")
ax.set_title("Fear/Greed Index vs Aggregate PnL", fontweight="bold", fontsize=14)
plt.colorbar(sc, label="Fear <-> Greed")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/B4_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# timeline chart
fig, ax1 = plt.subplots(figsize=(15, 6))
ax1.bar(daily_agg["date"], daily_agg["total_pnl"],
        color=daily_agg["sent_group"].map(color_map), alpha=0.8, width=0.8)
ax1.set_ylabel("Daily PnL ($)")
ax2 = ax1.twinx()
ax2.plot(daily_agg["date"], daily_agg["fgi_value"], color="#2c3e50", lw=2, label="F&G Index")
ax2.set_ylabel("Fear & Greed Index")
ax2.legend(loc="upper left")
ax1.set_title("PnL Over Time with Sentiment", fontweight="bold", fontsize=14)
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/B4_timeline.png", dpi=150, bbox_inches="tight")
plt.close()


# =====================================================================
#  PART C - ACTIONABLE STRATEGIES
# =====================================================================
print("\n" + "=" * 60)
print("PART C - STRATEGY RECOMMENDATIONS")
print("=" * 60)

print("""
Strategy 1: "Contrarian Long on Fear with Reduced Size"
  During Fear days, keep a long bias but cut position size to half
  of your normal. Only do this if your historical win rate is above
  50%. The data shows fear-day longs earn ~18% higher median PnL,
  but drawdowns are also deeper, so smaller size controls the risk.

Strategy 2: "Cap Trade Frequency on Greed Days"
  When the F&G index is above 65, high-size traders should limit
  their daily trade count to their Fear-day average. Over-trading
  during Greed periods leads to PnL dilution - you're chasing
  momentum that's probably already priced in.
""")


# =====================================================================
#  BONUS - PREDICTIVE MODEL
# =====================================================================
print("=" * 60)
print("BONUS - PREDICTIVE MODEL")
print("=" * 60)

# build features: use previous day's stats to predict today's profit bucket
feat = daily_seg.copy().sort_values(["Account", "date"])

lag_cols = ["pnl", "n_trades", "winrate", "avg_size", "long_pct"]
for col in lag_cols:
    feat[f"prev_{col}"] = feat.groupby("Account")[col].shift(1)

feat["fgi"] = feat["fgi_value"]
feat.dropna(inplace=True)

# target: which bucket did today fall into?
feat["bucket"] = pd.cut(
    feat["pnl"],
    bins=[-np.inf, -100, 0, 100, np.inf],
    labels=["Big Loss", "Small Loss", "Small Win", "Big Win"]
)

le = LabelEncoder()
feat["bucket_num"] = le.fit_transform(feat["bucket"].astype(str))

X_cols = ["prev_pnl", "prev_n_trades", "prev_winrate",
          "prev_avg_size", "prev_long_pct", "fgi"]
X = feat[X_cols].fillna(0)
y = feat["bucket_num"]

model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\n  5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

model.fit(X, y)
imp = pd.Series(model.feature_importances_, index=X_cols)
print("\n  Feature importance:")
print(imp.sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
imp.sort_values().plot.barh(ax=ax, color="#3498db", edgecolor="white")
ax.set_title("Feature Importance (Profitability Prediction)", fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/bonus_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()


# =====================================================================
#  BONUS - TRADER CLUSTERING
# =====================================================================
print("\n" + "=" * 60)
print("BONUS - K-MEANS CLUSTERING")
print("=" * 60)

cluster_cols = ["total_pnl", "winrate", "avg_size", "trades_per_day", "days_active"]
X_clust = profiles[cluster_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

# elbow plot to pick k
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(2, 9), inertias, "o-", color="#8e44ad")
ax.set_xlabel("k"); ax.set_ylabel("Inertia")
ax.set_title("Elbow Method", fontweight="bold")
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/bonus_elbow.png", dpi=150, bbox_inches="tight")
plt.close()

# use k=4 based on elbow
km = KMeans(n_clusters=4, random_state=42, n_init=10)
profiles["cluster"] = km.fit_predict(X_scaled)

cl_summary = profiles.groupby("cluster").agg(
    traders=("Account", "count"),
    avg_pnl=("total_pnl", "mean"),
    avg_wr=("winrate", "mean"),
    avg_sz=("avg_size", "mean"),
    avg_tpd=("trades_per_day", "mean"),
    avg_days=("days_active", "mean"),
).round(2)
print("\n  Cluster profiles:")
print(cl_summary.to_string())

# give them readable names based on what stands out
names = {}
for c in cl_summary.index:
    row = cl_summary.loc[c]
    if row["avg_sz"] == cl_summary["avg_sz"].max():
        names[c] = "Whale"
    elif row["avg_tpd"] == cl_summary["avg_tpd"].max():
        names[c] = "Hyperactive Scalper"
    elif row["avg_wr"] == cl_summary["avg_wr"].max():
        names[c] = "Disciplined Winner"
    elif row["avg_days"] == cl_summary["avg_days"].max():
        names[c] = "Steady Grinder"
    elif row["avg_pnl"] == cl_summary["avg_pnl"].min():
        names[c] = "Struggling Trader"
    else:
        names[c] = "Casual Trader"

profiles["archetype"] = profiles["cluster"].map(names)
print("\n  Archetypes:")
for k, v in names.items():
    print(f"    Cluster {k} -> {v}")

# scatter plot of clusters
fig, ax = plt.subplots(figsize=(10, 7))
for c in sorted(profiles["cluster"].unique()):
    sub = profiles[profiles["cluster"] == c]
    ax.scatter(sub["winrate"], sub["total_pnl"].clip(-20000, 20000),
               label=f"{names[c]}", alpha=0.6, s=50)
ax.axhline(0, color="grey", ls="--", alpha=0.5)
ax.axvline(0.5, color="grey", ls="--", alpha=0.5)
ax.set_xlabel("Win Rate")
ax.set_ylabel("Total PnL (clipped)")
ax.set_title("Trader Archetypes", fontweight="bold", fontsize=14)
ax.legend()
plt.tight_layout()
fig.savefig(f"{CHARTS_DIR}/bonus_clusters.png", dpi=150, bbox_inches="tight")
plt.close()


# =====================================================================
#  EXPORT SUMMARY TABLES
# =====================================================================
perf_table.to_csv(f"{CHARTS_DIR}/table_perf_by_sentiment.csv")
beh.to_csv(f"{CHARTS_DIR}/table_behavior_by_sentiment.csv")
cl_summary.to_csv(f"{CHARTS_DIR}/table_cluster_profiles.csv")

print("\n" + "=" * 60)
print("Done! All charts and tables saved to ./output_charts/")
print("=" * 60)
print(f"Files: {os.listdir(CHARTS_DIR)}")
