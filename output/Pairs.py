import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

# Function to identify pairs trades
def identify_pairs_trades(df, time_threshold_hours):
    df = df.sort_values(['rfqCounterparty', 'datetime'])
    pairs = []
    for _, group in df.groupby('rfqCounterparty'):
        for i in range(len(group) - 1):
            if (group.iloc[i+1]['datetime'] - group.iloc[i]['datetime']) <= timedelta(hours=time_threshold_hours):
                if group.iloc[i]['sym'] != group.iloc[i+1]['sym']:
                    pairs.append({
                        'counterparty': group.iloc[i]['rfqCounterparty'],
                        'datetime': group.iloc[i]['datetime'],
                        'sym1': min(group.iloc[i]['sym'], group.iloc[i+1]['sym']),
                        'sym2': max(group.iloc[i]['sym'], group.iloc[i+1]['sym']),
                        'price_shown1': pd.notnull(group.iloc[i]['price']),
                        'price_shown2': pd.notnull(group.iloc[i+1]['price']),
                        'traded1': group.iloc[i]['normalizedState'] == 'DONE',
                        'traded2': group.iloc[i+1]['normalizedState'] == 'DONE',
                    })
    return pd.DataFrame(pairs)

# Pairs Trading Dashboard
def pairs_trading_dashboard(df):
    st.header("Pairs Trading Analysis")

    # User input for time threshold
    time_threshold = st.slider("Time threshold for pairs (hours)", 1, 24, 12)

    # Identify pairs trades
    pairs_df = identify_pairs_trades(df, time_threshold)

    if pairs_df.empty:
        st.write("No pairs trades found within the specified time threshold.")
        return

    # 1. Most common pairs
    st.subheader("Most Common Pairs")
    pair_counts = pairs_df.groupby(['sym1', 'sym2']).size().sort_values(ascending=False).head(10)
    fig = px.bar(x=pair_counts.index.map(lambda x: f"{x[0]} - {x[1]}"), y=pair_counts.values,
                 labels={'x': 'Bond Pair', 'y': 'Count'},
                 title='Top 10 Most Traded Pairs')
    st.plotly_chart(fig)

    # 2. Common times for pairs trading
    st.subheader("Common Times for Pairs Trading")
    pairs_df['hour'] = pairs_df['datetime'].dt.hour
    hourly_counts = pairs_df['hour'].value_counts().sort_index()
    fig = px.line(x=hourly_counts.index, y=hourly_counts.values,
                  labels={'x': 'Hour of Day', 'y': 'Number of Pairs Trades'},
                  title='Pairs Trades by Hour of Day')
    st.plotly_chart(fig)

    # 3. Price shown and trading analysis
    st.subheader("Price Shown and Trading Analysis")
    price_shown = (pairs_df['price_shown1'] & pairs_df['price_shown2']).mean() * 100
    traded = (pairs_df['traded1'] & pairs_df['traded2']).mean() * 100
    st.write(f"Percentage of pairs where price was shown for both bonds: {price_shown:.2f}%")
    st.write(f"Percentage of pairs where both bonds were traded: {traded:.2f}%")

    # 4. Top counterparties in pairs trading
    st.subheader("Top Counterparties in Pairs Trading")
    top_counterparties = pairs_df['counterparty'].value_counts().head(10)
    fig = px.bar(x=top_counterparties.index, y=top_counterparties.values,
                 labels={'x': 'Counterparty', 'y': 'Number of Pairs Trades'},
                 title='Top 10 Counterparties in Pairs Trading')
    st.plotly_chart(fig)

    # 5. Pairs trading volume over time
    st.subheader("Pairs Trading Volume Over Time")
    pairs_df['date'] = pairs_df['datetime'].dt.date
    daily_volume = pairs_df.groupby('date').size()
    fig = px.line(x=daily_volume.index, y=daily_volume.values,
                  labels={'x': 'Date', 'y': 'Number of Pairs Trades'},
                  title='Daily Volume of Pairs Trades')
    st.plotly_chart(fig)

# In your Streamlit app, within tab 5
with tab5:
    pairs_trading_dashboard(df)
