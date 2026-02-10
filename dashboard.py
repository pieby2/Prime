"""
Streamlit dashboard for visualizing the sentiment vs trader performance analysis.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Trader vs Sentiment", layout="wide", page_icon="chart_with_upward_trend")

# some custom styling
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    [data-testid="stMetricValue"] {font-size: 1.3rem;}
    .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and prep data, mirroring the steps from analysis.py"""

    # Use absolute paths to ensure files are found regardless of CWD
    sent = pd.read_csv(os.path.join(SCRIPT_DIR, "fear_greed_index.csv"))
    trades = pd.read_csv(os.path.join(SCRIPT_DIR, "historical_data.csv"))

    sent["date"] = pd.to_datetime(sent["date"])
    sent.rename(columns={"value": "fgi_value", "classification": "sentiment"}, inplace=True)
    sent["sent_group"] = sent["sentiment"].map({
        "Extreme Fear": "Fear", "Fear": "Fear",
        "Neutral": "Neutral",
        "Greed": "Greed", "Extreme Greed": "Greed"
    })

    trades["Timestamp IST"] = pd.to_datetime(
        trades["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    trades["date"] = trades["Timestamp IST"].dt.normalize()

    for c in ["Closed PnL", "Fee", "Size USD", "Size Tokens", "Execution Price", "Start Position"]:
        trades[c] = pd.to_numeric(trades[c], errors="coerce")
    trades.dropna(subset=["date", "Closed PnL"], inplace=True)

    merged = trades.merge(sent[["date", "fgi_value", "sentiment", "sent_group"]],
                          on="date", how="inner")

    # closing trades only
    merged["is_close"] = merged["Direction"].str.contains("Close", case=False, na=False)
    closes = merged[merged["is_close"]].copy()
    closes["won"] = (closes["Closed PnL"] > 0).astype(int)

    daily = (
        closes.groupby(["date", "Account", "sent_group", "sentiment", "fgi_value"])
        .agg(pnl=("Closed PnL", "sum"), n_trades=("Closed PnL", "count"),
             n_wins=("won", "sum"), avg_size=("Size USD", "mean"),
             volume=("Size USD", "sum"))
        .reset_index()
    )
    daily["winrate"] = daily["n_wins"] / daily["n_trades"]

    # long/short ratio
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

    # per-account stats
    profiles = (
        closes.groupby("Account")
        .agg(total_pnl=("Closed PnL", "sum"), num_trades=("Closed PnL", "count"),
             winrate=("won", "mean"), avg_size=("Size USD", "mean"),
             days_active=("date", "nunique"))
        .reset_index()
    )
    profiles["trades_per_day"] = profiles["num_trades"] / profiles["days_active"]

    return merged, daily, profiles


# load everything
raw, daily, profiles = load_data()

# sidebar filters
st.sidebar.title("Filters")
st.sidebar.markdown("---")

sent_filter = st.sidebar.multiselect(
    "Sentiment", ["Fear", "Neutral", "Greed"],
    default=["Fear", "Neutral", "Greed"]
)

top_coins = raw["Coin"].value_counts().head(10).index.tolist()
coin_filter = st.sidebar.multiselect(
    "Coins (top 10)", top_coins, default=top_coins[:5]
)

date_min, date_max = raw["date"].min().date(), raw["date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(date_min, date_max),
                                    min_value=date_min, max_value=date_max)

# apply filters
mask = (
    raw["sent_group"].isin(sent_filter)
    & raw["Coin"].isin(coin_filter)
    & (raw["date"].dt.date >= date_range[0])
    & (raw["date"].dt.date <= date_range[1])
)
filt = raw[mask]
filt_daily = daily[
    daily["sent_group"].isin(sent_filter)
    & (daily["date"].dt.date >= date_range[0])
    & (daily["date"].dt.date <= date_range[1])
]

# header
st.title("Trader Performance vs Market Sentiment")
st.markdown("How does Bitcoin fear/greed affect Hyperliquid traders?")

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Trades", f"{len(filt):,}")
k2.metric("Unique Traders", filt["Account"].nunique())
k3.metric("Date Range", f"{(date_range[1] - date_range[0]).days} days")
avg_fgi = filt["fgi_value"].mean()
k4.metric("Avg F&G Index", f"{avg_fgi:.0f}" if not np.isnan(avg_fgi) else "N/A")

st.markdown("---")

# main content in tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Sentiment Impact", "Behavior", "Trader Segments", "Predictions"
])

colors = {"Fear": "#ef4444", "Neutral": "#94a3b8", "Greed": "#22c55e"}

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        perf = filt_daily.groupby("sent_group").agg(
            mean_pnl=("pnl", "mean"), med_pnl=("pnl", "median"),
            avg_wr=("winrate", "mean"), n=("pnl", "count")
        ).reset_index()

        fig = px.bar(perf, x="sent_group", y="mean_pnl", color="sent_group",
                     color_discrete_map=colors, title="Mean Daily PnL by Sentiment")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="PnL ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filt_daily, x="sent_group", y="winrate", color="sent_group",
                     color_discrete_map=colors, title="Win Rate Distribution")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Win Rate")
        st.plotly_chart(fig, use_container_width=True)

    # timeline with dual axis
    daily_global = (
        filt_daily.groupby(["date", "sent_group", "fgi_value"])
        .agg(total_pnl=("pnl", "sum")).reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_global["date"], y=daily_global["total_pnl"],
        marker_color=daily_global["sent_group"].map(colors),
        name="PnL"
    ))
    fig.add_trace(go.Scatter(
        x=daily_global["date"], y=daily_global["fgi_value"],
        yaxis="y2", name="F&G Index", line=dict(color="#f59e0b", width=2)
    ))
    fig.update_layout(
        title="PnL Timeline with Sentiment Overlay",
        yaxis=dict(title="PnL ($)"),
        yaxis2=dict(title="F&G Index", overlaying="y", side="right", range=[0, 100]),
        hovermode="x unified", height=450
    )
    st.plotly_chart(fig, use_container_width=True)


with tab2:
    col1, col2 = st.columns(2)

    with col1:
        beh = filt_daily.groupby("sent_group").agg(
            avg_trades=("n_trades", "mean"), avg_size=("avg_size", "mean")
        ).reset_index()
        fig = px.bar(beh, x="sent_group", y="avg_trades", color="sent_group",
                     color_discrete_map=colors, title="Avg Trade Count")
        fig.update_layout(showlegend=False, yaxis_title="Trades/Day")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(beh, x="sent_group", y="avg_size", color="sent_group",
                     color_discrete_map=colors, title="Avg Trade Size")
        fig.update_layout(showlegend=False, yaxis_title="Size ($)")
        st.plotly_chart(fig, use_container_width=True)

    # coin x sentiment heatmap
    pivot = filt.groupby(["Coin", "sent_group"])["Closed PnL"].mean().unstack(fill_value=0)
    # show top 15 coins by volume
    top15 = filt.groupby("Coin")["Size USD"].sum().nlargest(15).index
    pivot = pivot.loc[pivot.index.isin(top15)]

    if not pivot.empty:
        fig = px.imshow(pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                        color_continuous_scale="RdYlGn", title="Mean PnL: Coin x Sentiment",
                        labels=dict(color="Mean PnL ($)"))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


with tab3:
    # trader scatter
    fig = px.scatter(
        profiles, x="winrate",
        y=profiles["total_pnl"].clip(-30000, 30000),
        size=profiles["avg_size"].clip(0, 5000),
        color="trades_per_day", color_continuous_scale="Viridis",
        hover_data=["Account", "num_trades"],
        title="Trader Landscape (size = avg trade, color = frequency)",
        labels={"y": "Total PnL (clipped)", "color": "Trades/Day"}
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    # top traders table
    st.subheader("Top 10 Traders by PnL")
    top = profiles.nlargest(10, "total_pnl")[
        ["Account", "total_pnl", "num_trades", "winrate", "avg_size", "days_active"]
    ].copy()
    top["Account"] = top["Account"].str[:10] + "..."
    top.columns = ["Account", "Total PnL", "Trades", "Win Rate", "Avg Size", "Days Active"]
    st.dataframe(top, use_container_width=True)


with tab4:
    st.subheader("Gradient Boosting - Next-Day Profitability Prediction")
    st.markdown("""
    A Gradient Boosting model was trained to predict which profitability
    bucket a trader will fall into the next day, using lagged behavioral
    features and the current F&G index. **5-fold CV accuracy: ~71%**.
    """)

    # show pre-generated charts if available
    for fname, caption in [
        ("bonus_feature_importance.png", "Feature Importance"),
        ("bonus_clusters.png", "Trader Archetypes (K-Means)")
    ]:
        path = os.path.join(SCRIPT_DIR, "output_charts", fname)
        if os.path.exists(path):
            st.image(path, caption=caption)
        else:
            st.info(f"Run analysis.py first to generate {fname}")

    st.markdown("---")
    st.markdown("*Built for the Primetrade.ai Data Science Intern assignment*")
