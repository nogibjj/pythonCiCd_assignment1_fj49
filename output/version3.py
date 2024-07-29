import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from itertools import combinations


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dummy_rfq_data_large.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


df = load_data()

st.title("Comprehensive RFQ Analysis Dashboard")

# Sidebar for threshold inputs
st.sidebar.header("Threshold Settings")
time_threshold = st.sidebar.slider("Time Threshold (hours)", 1, 48, 12)
min_rfqs = st.sidebar.slider("Minimum Number of RFQs", 1, 10, 2)


# Function to categorize RFQs
# def categorize_rfqs(df, time_threshold, min_rfqs):
#     df_sorted = df.sort_values(["rfqCounterparty", "datetime"]).reset_index(drop=True)
#     df_sorted["time_diff"] = df_sorted.groupby("rfqCounterparty")["datetime"].diff()

#     # Identify potential pairs
#     df_sorted["pair_group"] = df_sorted.groupby("rfqCounterparty")[
#         "time_diff"
#     ].transform(lambda x: (x > timedelta(hours=time_threshold)).cumsum())

#     # Count RFQs for each counterparty-symbol combination
#     df_sorted["rfq_count"] = (
#         df_sorted.groupby(["rfqCounterparty", "sym"]).cumcount() + 1
#     )

#     # Count unique symbols in each pair group
#     symbol_counts = df_sorted.groupby(["rfqCounterparty", "pair_group"])[
#         "sym"
#     ].nunique()
#     pair_counts = df_sorted.groupby(["rfqCounterparty", "pair_group"]).size()

#     # Identify pairs trading
#     pairs_trading = (symbol_counts >= 2) & (pair_counts >= min_rfqs)
#     pairs_trading_groups = pairs_trading[pairs_trading].index

#     # Create a mapping for pairs trading groups
#     df_sorted["pairs_mapping"] = False
#     for counterparty, group in pairs_trading_groups:
#         mask = (df_sorted["rfqCounterparty"] == counterparty) & (
#             df_sorted["pair_group"] == group
#         )
#         df_sorted.loc[mask, "pairs_mapping"] = True

#     # Categorize
#     conditions = [
#         df_sorted["pairs_mapping"],
#         (df_sorted["time_diff"] <= timedelta(hours=time_threshold))
#         & (df_sorted["rfq_count"] >= min_rfqs)
#         & (
#             df_sorted["rfqL0VerbStrid"]
#             != df_sorted.groupby(["rfqCounterparty", "sym"])["rfqL0VerbStrid"].shift(1)
#         ),
#         (df_sorted["time_diff"] <= timedelta(hours=time_threshold))
#         & (df_sorted["rfq_count"] >= min_rfqs),
#     ]
#     choices = ["Pairs Trading", "Pattern Day Trading", "VWAPER"]
#     df_sorted["category"] = np.select(conditions, choices, default="One-off")


#     return df_sorted
import pandas as pd
from datetime import timedelta


#### handles 2, works fine
def categorize_rfqs(df, time_threshold_hours):
    df = df.sort_values(["rfqCounterparty", "sym", "datetime"]).reset_index(drop=True)
    df["category"] = "One-off"

    for (counterparty, sym), group in df.groupby(["rfqCounterparty", "sym"]):
        if len(group) > 1:
            for i in range(1, len(group)):
                if (
                    group.iloc[i]["datetime"] - group.iloc[i - 1]["datetime"]
                ) <= timedelta(hours=time_threshold_hours):
                    if (
                        group.iloc[i]["rfqL0VerbStrid"]
                        == group.iloc[i - 1]["rfqL0VerbStrid"]
                    ):
                        df.loc[group.index[i - 1 : i + 1], "category"] = "VWAPER"
                    else:
                        df.loc[group.index[i - 1 : i + 1], "category"] = (
                            "Pattern Day Trading"
                        )

    return df


# def categorize_rfqs(df, time_threshold_hours, min_rfqs):
#     df = df.sort_values(["rfqCounterparty", "sym", "datetime"]).reset_index(drop=True)
#     df["category"] = "One-off"

#     for (counterparty, sym), group in df.groupby(["rfqCounterparty", "sym"]):
#         if len(group) >= min_rfqs:
#             group = group.sort_values("datetime")
#             for i in range(min_rfqs - 1, len(group)):
#                 if (
#                     group.iloc[i]["datetime"] - group.iloc[i - min_rfqs + 1]["datetime"]
#                 ) <= timedelta(hours=time_threshold_hours):
#                     if (
#                         group.iloc[i - min_rfqs + 1 : i + 1]["rfqL0VerbStrid"].nunique()
#                         == 1
#                     ):
#                         df.loc[group.index[i - min_rfqs + 1 : i + 1], "category"] = (
#                             "VWAPER"
#                         )
#                     else:
#                         df.loc[group.index[i - min_rfqs + 1 : i + 1], "category"] = (
#                             "Pattern Day Trading"
#                         )

#     return df


# Example usage:
# df = pd.read_csv('your_data.csv')
# df['datetime'] = pd.to_datetime(df['datetime'])
# categorized_df = categorize_rfqs(df, time_threshold_hours=12)


categorized_df = categorize_rfqs(df, time_threshold_hours=time_threshold)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Basic Analysis",
        "One-off Inquiries",
        "VWAPER",
        "Pattern Day Trading",
        "Pairs Trading",
    ]
)

with tab1:
    st.header("Basic Descriptive Analysis")

    st.subheader("Dataset Overview")
    st.write(f"Total RFQs: {len(df)}")
    st.write(
        f"Date Range: {df['datetime'].min().date()} to {df['datetime'].max().date()}"
    )
    st.write(f"Unique Securities: {df['sym'].nunique()}")
    st.write(f"Unique Counterparties: {df['rfqCounterparty'].nunique()}")

    st.subheader("RFQ Categorization")
    category_counts = categorized_df["category"].value_counts()
    st.write(category_counts)
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Distribution of RFQ Categories",
    )
    st.plotly_chart(fig)

    st.subheader("Daily RFQ Volume")
    daily_volume = df.resample("D", on="datetime").size()
    fig = px.line(
        x=daily_volume.index,
        y=daily_volume.values,
        labels={"x": "Date", "y": "Number of RFQs"},
    )
    st.plotly_chart(fig)

    st.subheader("Top 10 Securities by RFQ Count")
    top_securities = df["sym"].value_counts().head(10)
    fig = px.bar(
        x=top_securities.index,
        y=top_securities.values,
        labels={"x": "Security", "y": "Number of RFQs"},
    )
    st.plotly_chart(fig)

    st.subheader("Top 10 Counterparties by RFQ Count")
    top_counterparties = df["rfqCounterparty"].value_counts().head(10)
    fig = px.bar(
        x=top_counterparties.index,
        y=top_counterparties.values,
        labels={"x": "Counterparty", "y": "Number of RFQs"},
    )
    st.plotly_chart(fig)

with tab2:
    st.header("One-off Inquiries Analysis")
    one_off_df = categorized_df[categorized_df["category"] == "One-off"]

    st.subheader("Descriptive Statistics for One-off Inquiries")
    st.write(one_off_df[["rfqL0DealQty", "liquidityScore"]].describe())

    st.subheader("Top Securities in One-off Inquiries")
    top_securities = one_off_df["sym"].value_counts().head(10)
    fig = px.bar(
        x=top_securities.index,
        y=top_securities.values,
        labels={"x": "Security", "y": "Count"},
    )
    st.plotly_chart(fig)

    st.subheader("Buy vs Sell Distribution in One-off Inquiries")
    buy_sell_dist = one_off_df["rfqL0VerbStrid"].value_counts()
    fig = px.pie(values=buy_sell_dist.values, names=buy_sell_dist.index)
    st.plotly_chart(fig)


# with tab3:
#     st.header("VWAPER Analysis")
#     vwaper_df = categorized_df[categorized_df["category"] == "VWAPER"]

#     st.subheader("Overall VWAPER Statistics")
#     st.write(f"Total VWAPER Trades: {len(vwaper_df)}")
#     st.write(
#         f"Unique Counterparties Involved: {vwaper_df['rfqCounterparty'].nunique()}"
#     )
#     st.write(f"Unique Securities: {vwaper_df['sym'].nunique()}")

#     st.subheader("Distribution of VWAPER Trades")
#     vwaper_counts = (
#         vwaper_df.groupby(["rfqCounterparty", "sym"])
#         .size()
#         .reset_index(name="trade_count")
#     )
#     fig = px.histogram(
#         vwaper_counts,
#         x="trade_count",
#         labels={
#             "trade_count": f"Number of VWAPER Trades (min {min_rfqs})",
#             "count": "Frequency",
#         },
#         title=f"Distribution of VWAPER Trade Counts (min {min_rfqs} trades)",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Top Counterparties with VWAPER Behavior")
#     top_vwaper_counterparties = vwaper_df["rfqCounterparty"].value_counts().head(10)
#     fig = px.bar(
#         x=top_vwaper_counterparties.index,
#         y=top_vwaper_counterparties.values,
#         labels={"x": "Counterparty", "y": "Number of VWAPER Trades"},
#         title="Top 10 Counterparties in VWAPER Trading",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Most Frequent VWAPER Securities")
#     vwaper_securities = vwaper_df["sym"].value_counts().head(10)
#     fig = px.bar(
#         x=vwaper_securities.index,
#         y=vwaper_securities.values,
#         labels={"x": "Security", "y": "Number of VWAPER Trades"},
#         title="Top 10 Securities in VWAPER Trading",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Detailed Counterparty Analysis")
#     selected_counterparty = st.selectbox(
#         "Select a counterparty for detailed analysis:",
#         vwaper_df["rfqCounterparty"].unique(),
#     )
#     counterparty_data = vwaper_df[vwaper_df["rfqCounterparty"] == selected_counterparty]

#     st.write(f"Analysis for {selected_counterparty}")
#     counterparty_securities = counterparty_data["sym"].value_counts().head(5)
#     st.write(f"Top 5 securities (with at least {min_rfqs} trades):")
#     st.write(counterparty_securities)

#     st.write("Volume analysis:")
#     volume_analysis = counterparty_data.groupby("sym")["rfqL0DealQty"].agg(
#         ["mean", "min", "max"]
#     )
#     st.write(volume_analysis)

#     st.subheader("Time Between VWAPER Trades")
#     counterparty_data = counterparty_data.sort_values("datetime")
#     counterparty_data["time_between_trades"] = (
#         counterparty_data.groupby("sym")["datetime"].diff().dt.total_seconds() / 3600
#     )  # in hours
#     fig = px.box(
#         counterparty_data,
#         x="sym",
#         y="time_between_trades",
#         labels={"sym": "Security", "time_between_trades": "Hours Between Trades"},
#         title=f"Time Between VWAPER Trades for {selected_counterparty}",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Recent VWAPER Trades")
#     st.write(
#         counterparty_data.sort_values("datetime", ascending=False).head(10)[
#             ["datetime", "sym", "rfqL0DealQty", "rfqL0VerbStrid"]
#         ]
#     )
# for 2, works fine
with tab3:
    st.header("VWAPER Analysis")
    vwaper_df = categorized_df[categorized_df["category"] == "VWAPER"]

    st.subheader("Overall VWAPER Statistics")
    st.write(f"Total VWAPER Trades: {len(vwaper_df)}")
    st.write(
        f"Unique Counterparties Involved: {vwaper_df['rfqCounterparty'].nunique()}"
    )
    st.write(f"Unique Securities: {vwaper_df['sym'].nunique()}")

    st.subheader("Distribution of VWAPER Trades")
    vwaper_counts = (
        vwaper_df.groupby(["rfqCounterparty", "sym"])
        .size()
        .reset_index(name="trade_count")
    )
    fig = px.histogram(
        vwaper_counts,
        x="trade_count",
        labels={"trade_count": "Number of VWAPER Trades", "count": "Frequency"},
        title="Distribution of VWAPER Trade Counts",
    )
    st.plotly_chart(fig)

    st.subheader("Top Counterparties with VWAPER Behavior")
    top_vwaper_counterparties = vwaper_df["rfqCounterparty"].value_counts().head(10)
    fig = px.bar(
        x=top_vwaper_counterparties.index,
        y=top_vwaper_counterparties.values,
        labels={"x": "Counterparty", "y": "Number of VWAPER Trades"},
        title="Top 10 Counterparties in VWAPER Trading",
    )
    st.plotly_chart(fig)

    st.subheader("Most Frequent VWAPER Securities")
    vwaper_securities = vwaper_df["sym"].value_counts().head(10)
    fig = px.bar(
        x=vwaper_securities.index,
        y=vwaper_securities.values,
        labels={"x": "Security", "y": "Number of VWAPER Trades"},
        title="Top 10 Securities in VWAPER Trading",
    )
    st.plotly_chart(fig)

    st.subheader("Detailed Counterparty Analysis")
    selected_counterparty = st.selectbox(
        "Select a counterparty for detailed analysis:",
        vwaper_df["rfqCounterparty"].unique(),
    )
    counterparty_data = vwaper_df[vwaper_df["rfqCounterparty"] == selected_counterparty]

    st.write(f"Analysis for {selected_counterparty}")
    counterparty_securities = counterparty_data["sym"].value_counts().head(5)
    st.write("Top 5 securities:")
    st.write(counterparty_securities)

    st.write("Volume analysis:")
    volume_analysis = counterparty_data.groupby("sym")["rfqL0DealQty"].agg(
        ["mean", "min", "max"]
    )
    st.write(volume_analysis)

    st.subheader("Time Between VWAPER Trades")
    counterparty_data = counterparty_data.sort_values("datetime")
    counterparty_data["time_between_trades"] = (
        counterparty_data.groupby("sym")["datetime"].diff().dt.total_seconds() / 3600
    )  # in hours
    fig = px.box(
        counterparty_data,
        x="sym",
        y="time_between_trades",
        labels={"sym": "Security", "time_between_trades": "Hours Between Trades"},
        title=f"Time Between VWAPER Trades for {selected_counterparty}",
    )
    st.plotly_chart(fig)

    st.subheader("Recent VWAPER Trades")
    st.write(
        counterparty_data.sort_values("datetime", ascending=False)[
            ["datetime", "sym", "rfqL0DealQty", "rfqL0VerbStrid"]
        ]
    )


# with tab4:
#     st.header("Pattern Day Trading Analysis")
#     pdt_df = categorized_df[categorized_df["category"] == "Pattern Day Trading"]

#     st.subheader("Overall Pattern Day Trading Statistics")
#     st.write(f"Total Pattern Day Trades: {len(pdt_df)}")
#     st.write(f"Unique Counterparties Involved: {pdt_df['rfqCounterparty'].nunique()}")
#     st.write(f"Unique Symbols Traded: {pdt_df['sym'].nunique()}")

#     st.subheader("Top Counterparties Engaging in Pattern Day Trading")
#     top_pdt_counterparties = pdt_df["rfqCounterparty"].value_counts().head(10)
#     fig = px.bar(
#         x=top_pdt_counterparties.index,
#         y=top_pdt_counterparties.values,
#         labels={"x": "Counterparty", "y": "Number of Pattern Day Trades"},
#         title="Top 10 Counterparties in Pattern Day Trading",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Most Common Symbols in Pattern Day Trading")
#     pdt_symbols = pdt_df["sym"].value_counts().head(10)
#     fig = px.bar(
#         x=pdt_symbols.index,
#         y=pdt_symbols.values,
#         labels={"x": "Symbol", "y": "Number of Pattern Day Trades"},
#         title="Top 10 Symbols in Pattern Day Trading",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Volume Distribution in Pattern Day Trading")
#     fig = px.histogram(
#         pdt_df,
#         x="rfqL0DealQty",
#         nbins=50,
#         labels={"rfqL0DealQty": "Deal Quantity", "count": "Frequency"},
#         title="Distribution of Deal Quantities in Pattern Day Trading",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Time Between Buy and Sell in Pattern Day Trading")
#     pdt_df["time_between_trades"] = (
#         pdt_df["time_diff"].dt.total_seconds() / 3600
#     )  # Convert to hours
#     fig = px.histogram(
#         pdt_df,
#         x="time_between_trades",
#         nbins=50,
#         labels={
#             "time_between_trades": "Time Between Trades (hours)",
#             "count": "Frequency",
#         },
#         title="Distribution of Time Between Buy and Sell",
#     )
#     st.plotly_chart(fig)

#     st.subheader("Detailed Pattern Day Trading Analysis")
#     selected_pdt_counterparty = st.selectbox(
#         "Select a counterparty for detailed PDT analysis:",
#         pdt_df["rfqCounterparty"].unique(),
#     )
#     counterparty_pdt_data = pdt_df[
#         pdt_df["rfqCounterparty"] == selected_pdt_counterparty
#     ]

#     st.write(f"Analysis for {selected_pdt_counterparty}")
#     counterparty_pdt_symbols = counterparty_pdt_data["sym"].value_counts().head(5)
#     st.write("Top 5 symbols traded:")
#     st.write(counterparty_pdt_symbols)

#     st.write("Average volumes:")
#     avg_volumes = (
#         counterparty_pdt_data.groupby("sym")["rfqL0DealQty"]
#         .mean()
#         .sort_values(ascending=False)
#         .head(5)
#     )
#     st.write(avg_volumes)

#     st.write("Average time between trades:")
#     avg_time = (
#         counterparty_pdt_data.groupby("sym")["time_between_trades"]
#         .mean()
#         .sort_values(ascending=False)
#         .head(5)
#     )
#     st.write(avg_time)

with tab4:
    st.header("Pattern Day Trading Analysis")
    pdt_df = categorized_df[categorized_df["category"] == "Pattern Day Trading"]

    st.subheader("Overall Pattern Day Trading Statistics")
    st.write(f"Total Pattern Day Trades: {len(pdt_df)}")
    st.write(f"Unique Counterparties Involved: {pdt_df['rfqCounterparty'].nunique()}")
    st.write(f"Unique Symbols Traded: {pdt_df['sym'].nunique()}")

    st.subheader("Top Counterparties Engaging in Pattern Day Trading")
    top_pdt_counterparties = pdt_df["rfqCounterparty"].value_counts().head(10)
    fig = px.bar(
        x=top_pdt_counterparties.index,
        y=top_pdt_counterparties.values,
        labels={"x": "Counterparty", "y": "Number of Pattern Day Trades"},
        title="Top 10 Counterparties in Pattern Day Trading",
    )
    st.plotly_chart(fig)

    st.subheader("Most Common Symbols in Pattern Day Trading")
    pdt_symbols = pdt_df["sym"].value_counts().head(10)
    fig = px.bar(
        x=pdt_symbols.index,
        y=pdt_symbols.values,
        labels={"x": "Symbol", "y": "Number of Pattern Day Trades"},
        title="Top 10 Symbols in Pattern Day Trading",
    )
    st.plotly_chart(fig)

    st.subheader("Volume Distribution in Pattern Day Trading")
    fig = px.histogram(
        pdt_df,
        x="rfqL0DealQty",
        nbins=50,
        labels={"rfqL0DealQty": "Deal Quantity", "count": "Frequency"},
        title="Distribution of Deal Quantities in Pattern Day Trading",
    )
    st.plotly_chart(fig)

    st.subheader("Time Between Buy and Sell in Pattern Day Trading")
    pdt_df_sorted = pdt_df.sort_values(["rfqCounterparty", "sym", "datetime"])
    pdt_df_sorted["time_between_trades"] = (
        pdt_df_sorted.groupby(["rfqCounterparty", "sym"])["datetime"]
        .diff()
        .dt.total_seconds()
        / 3600
    )  # Convert to hours
    fig = px.histogram(
        pdt_df_sorted,
        x="time_between_trades",
        nbins=50,
        labels={
            "time_between_trades": "Time Between Trades (hours)",
            "count": "Frequency",
        },
        title="Distribution of Time Between Buy and Sell",
    )
    st.plotly_chart(fig)

    st.subheader("Detailed Pattern Day Trading Analysis")
    selected_pdt_counterparty = st.selectbox(
        "Select a counterparty for detailed PDT analysis:",
        pdt_df["rfqCounterparty"].unique(),
    )
    counterparty_pdt_data = pdt_df[
        pdt_df["rfqCounterparty"] == selected_pdt_counterparty
    ]

    st.write(f"Analysis for {selected_pdt_counterparty}")
    counterparty_pdt_symbols = counterparty_pdt_data["sym"].value_counts().head(5)
    st.write("Top 5 symbols traded:")
    st.write(counterparty_pdt_symbols)

    st.write("Average volumes:")
    avg_volumes = (
        counterparty_pdt_data.groupby("sym")["rfqL0DealQty"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    st.write(avg_volumes)

    st.write("Average time between trades:")
    counterparty_pdt_data_sorted = counterparty_pdt_data.sort_values(
        ["sym", "datetime"]
    )
    counterparty_pdt_data_sorted["time_between_trades"] = (
        counterparty_pdt_data_sorted.groupby("sym")["datetime"]
        .diff()
        .dt.total_seconds()
        / 3600
    )
    avg_time = (
        counterparty_pdt_data_sorted.groupby("sym")["time_between_trades"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )
    st.write(avg_time)

    st.subheader("Recent Pattern Day Trades")
    st.write(
        counterparty_pdt_data.sort_values("datetime", ascending=False).head(10)[
            ["datetime", "sym", "rfqL0DealQty", "rfqL0VerbStrid"]
        ]
    )

with tab5:
    st.header("Pairs Trading Analysis")
    # pairs_df = categorized_df[categorized_df["category"] == "Pairs Trading"]

    # st.subheader("Overall Pairs Trading Statistics")
    # st.write(f"Total Pairs Trades: {len(pairs_df)}")
    # st.write(f"Unique Counterparties Involved: {pairs_df['rfqCounterparty'].nunique()}")

    # st.subheader("Top Counterparties Engaging in Pairs Trading")
    # top_pairs_counterparties = pairs_df["rfqCounterparty"].value_counts().head(10)
    # fig = px.bar(
    #     x=top_pairs_counterparties.index,
    #     y=top_pairs_counterparties.values,
    #     labels={"x": "Counterparty", "y": "Number of Pairs Trades"},
    #     title="Top 10 Counterparties in Pairs Trading",
    # )
    # st.plotly_chart(fig)

    # st.subheader("Most Common Symbol Pairs")

    # def get_symbol_pairs(group):
    #     symbols = group["sym"].unique()
    #     return list(combinations(sorted(symbols), 2))

    # symbol_pairs = pairs_df.groupby(["rfqCounterparty", "pair_group"]).apply(
    #     get_symbol_pairs
    # )
    # all_pairs = [pair for pairs in symbol_pairs for pair in pairs]
    # pair_counts = pd.Series(all_pairs).value_counts()

    # fig = px.bar(
    #     x=[f"{pair[0]} - {pair[1]}" for pair in pair_counts.index[:10]],
    #     y=pair_counts.values[:10],
    #     labels={"x": "Symbol Pair", "y": "Frequency"},
    #     title="Top 10 Symbol Pairs in Pairs Trading",
    # )
    # st.plotly_chart(fig)

    # st.subheader("Detailed Pairs Trading Analysis")
    # selected_pairs_counterparty = st.selectbox(
    #     "Select a counterparty for detailed pairs analysis:",
    #     pairs_df["rfqCounterparty"].unique(),
    # )
    # counterparty_pairs_data = pairs_df[
    #     pairs_df["rfqCounterparty"] == selected_pairs_counterparty
    # ]

    # st.write(f"Analysis for {selected_pairs_counterparty}")
    # counterparty_symbol_pairs = counterparty_pairs_data.groupby("pair_group").apply(
    #     get_symbol_pairs
    # )
    # counterparty_pair_counts = pd.Series(
    #     [pair for pairs in counterparty_symbol_pairs for pair in pairs]
    # ).value_counts()

    # st.write("Top 5 symbol pairs:")
    # st.write(counterparty_pair_counts.head())

    # st.write("Average volumes for top pairs:")
    # for pair in counterparty_pair_counts.head().index:
    #     volumes = counterparty_pairs_data[counterparty_pairs_data["sym"].isin(pair)][
    #         "rfqL0DealQty"
    #     ]
    #     st.write(f"{pair[0]} - {pair[1]}: {volumes.mean():.2f}")

    # st.write("Average time between trades in a pair:")
    # counterparty_pairs_data["time_between_trades"] = (
    #     counterparty_pairs_data.groupby("pair_group")["datetime"]
    #     .diff()
    #     .dt.total_seconds()
    #     / 3600
    # )
    # avg_time = counterparty_pairs_data.groupby("pair_group")[
    #     "time_between_trades"
    # ].mean()
    # st.write(avg_time.describe())

    # st.subheader("Pair Trading Patterns")
    # fig = px.scatter(
    #     counterparty_pairs_data,
    #     x="datetime",
    #     y="rfqL0DealQty",
    #     color="sym",
    #     hover_data=["rfqL0VerbStrid"],
    #     title=f"Pair Trading Pattern for {selected_pairs_counterparty}",
    # )
    # st.plotly_chart(fig)

    # st.subheader("Correlation between Pair Volumes")
    # pivot_volumes = counterparty_pairs_data.pivot(
    #     index="datetime", columns="sym", values="rfqL0DealQty"
    # )
    # correlation_matrix = pivot_volumes.corr()
    # fig = px.imshow(
    #     correlation_matrix,
    #     labels=dict(color="Correlation"),
    #     title=f"Volume Correlation Matrix for {selected_pairs_counterparty}",
    # )
    # st.plotly_chart(fig)
