import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from collections import defaultdict

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('your_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

df = load_data()

st.title('Advanced Bond Trading RFQ Analysis Dashboard')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Counterparty Analysis", "Trading Strategies", "Liquidity Analysis", "Time-based Patterns"])

with tab1:
    st.header('Overview')

    # Basic stats
    st.subheader('Dataset Statistics')
    st.write(f"Total RFQs: {len(df)}")
    st.write(f"Date Range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    st.write(f"Unique Securities: {df['sym'].nunique()}")
    st.write(f"Unique Counterparties: {df['rfqCounterparty'].nunique()}")

    # RFQ Frequency
    st.subheader('RFQ Frequency Over Time')
    daily_rfq = df.resample('D', on='datetime').size()
    fig = px.line(x=daily_rfq.index, y=daily_rfq.values, labels={'x': 'Date', 'y': 'Number of RFQs'})
    st.plotly_chart(fig)

    # Top Securities
    st.subheader('Top 10 Most Queried Securities')
    top_securities = df['sym'].value_counts().head(10)
    fig = px.bar(x=top_securities.index, y=top_securities.values, labels={'x': 'Security', 'y': 'Number of RFQs'})
    st.plotly_chart(fig)

    # Buy vs Sell Distribution
    st.subheader('Buy vs Sell Distribution')
    buy_sell = df['rfqL0VerbStrid'].value_counts()
    fig = px.pie(values=buy_sell.values, names=buy_sell.index)
    st.plotly_chart(fig)

with tab2:
    st.header('Detailed Counterparty Analysis')

    # Time-based filtering
    col1, col2 = st.columns(2)
    with col1:
        time_diff = st.slider('Time Difference for Repeated Trades (hours)', 1, 168, 24)
    with col2:
        num_days = st.slider('Number of Days to Analyze', 1, 30, 7)

    end_date = df['datetime'].max()
    start_date = end_date - timedelta(days=num_days)
    filtered_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    def find_repeated_trades(df, time_threshold):
        df_sorted = df.sort_values(['rfqCounterparty', 'datetime'])
        df_sorted['time_diff'] = df_sorted.groupby('rfqCounterparty')['datetime'].diff()
        repeated = df_sorted[df_sorted['time_diff'] <= timedelta(hours=time_threshold)]
        return repeated

    repeated_trades = find_repeated_trades(filtered_df, time_diff)
    repeated_counterparties = repeated_trades['rfqCounterparty'].unique()

    st.subheader('Overview of Repeated Trading Activity')
    st.write(f"Number of counterparties with repeated trades within {time_diff} hours: {len(repeated_counterparties)}")
    st.write(f"Total number of repeated trades: {len(repeated_trades)}")

    # Tier Analysis of Repeated Traders
    tier_counts = repeated_trades['rfqCustTierInfo'].value_counts()
    fig = px.pie(values=tier_counts.values, names=tier_counts.index, title='Customer Tiers of Repeated Traders')
    st.plotly_chart(fig)

    # Liquidity Score Analysis
    st.subheader('Liquidity Score Analysis for Repeated Trades')
    avg_liquidity = repeated_trades.groupby('rfqCounterparty')['liquidityScore'].mean().sort_values(ascending=False)
    fig = px.bar(x=avg_liquidity.index, y=avg_liquidity.values, labels={'x': 'Counterparty', 'y': 'Average Liquidity Score'},
                 title='Average Liquidity Score for Repeated Traders')
    st.plotly_chart(fig)

    # Individual Counterparty Analysis
    st.subheader('Individual Counterparty Analysis')
    selected_counterparty = st.selectbox('Select a counterparty for detailed analysis:', repeated_counterparties)
    
    counterparty_trades = repeated_trades[repeated_trades['rfqCounterparty'] == selected_counterparty]
    
    st.write(f"Total repeated trades for {selected_counterparty}: {len(counterparty_trades)}")
    st.write(f"Customer Tier: {counterparty_trades['rfqCustTierInfo'].iloc[0]}")
    st.write(f"Average Liquidity Score: {counterparty_trades['liquidityScore'].mean():.2f}")
    
    # Bond Preference
    bond_preference = counterparty_trades['sym'].value_counts()
    fig = px.bar(x=bond_preference.index, y=bond_preference.values, labels={'x': 'Bond', 'y': 'Number of Repeated Trades'},
                 title=f'Bond Preference for {selected_counterparty}')
    st.plotly_chart(fig)
    
    # Trade Direction
    direction_counts = counterparty_trades['rfqL0VerbStrid'].value_counts()
    fig = px.pie(values=direction_counts.values, names=direction_counts.index, title=f'Trade Direction for {selected_counterparty}')
    st.plotly_chart(fig)
    
    # Trade Size Distribution
    fig = px.histogram(counterparty_trades, x='rfqL0DealQty', nbins=20, labels={'rfqL0DealQty': 'Deal Quantity'},
                       title=f'Trade Size Distribution for {selected_counterparty}')
    st.plotly_chart(fig)

    # Recent Trade History
    st.subheader(f"Recent Trade History for {selected_counterparty}")
    st.write(counterparty_trades[['datetime', 'sym', 'rfqL0DealQty', 'rfqL0VerbStrid', 'liquidityScore']].sort_values('datetime', ascending=False).head(10))

with tab3:
    st.header('Trading Strategies Analysis')

    # Pairs Trading Detection
    st.subheader('Potential Pairs Trading Detection')
    def detect_pairs_trading(df, time_threshold=timedelta(minutes=30)):
        pairs = defaultdict(int)
        for _, group in df.groupby('rfqCounterparty'):
            group = group.sort_values('datetime')
            for i in range(len(group) - 1):
                if group.iloc[i+1]['datetime'] - group.iloc[i]['datetime'] <= time_threshold:
                    pair = tuple(sorted([group.iloc[i]['sym'], group.iloc[i+1]['sym']]))
                    pairs[pair] += 1
        return pairs

    potential_pairs = detect_pairs_trading(filtered_df)
    top_pairs = sorted(potential_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    
    pair_names = [f"{pair[0][0]} - {pair[0][1]}" for pair in top_pairs]
    pair_counts = [pair[1] for pair in top_pairs]
    
    fig = px.bar(x=pair_names, y=pair_counts, labels={'x': 'Security Pair', 'y': 'Frequency'},
                 title='Top 10 Potential Pairs Trading Combinations')
    st.plotly_chart(fig)

    # Ladder Trading Analysis
    st.subheader('Potential Ladder Trading Detection')
    def detect_ladder_trading(df, time_threshold=timedelta(hours=24), min_rungs=3):
        ladder_trades = defaultdict(list)
        for _, group in df.groupby('rfqCounterparty'):
            group = group.sort_values('datetime')
            current_ladder = []
            for i in range(len(group)):
                if not current_ladder or (group.iloc[i]['datetime'] - current_ladder[-1]['datetime'] <= time_threshold):
                    current_ladder.append(group.iloc[i])
                else:
                    if len(current_ladder) >= min_rungs:
                        ladder_trades[group.iloc[i]['rfqCounterparty']].append(current_ladder)
                    current_ladder = [group.iloc[i]]
            if len(current_ladder) >= min_rungs:
                ladder_trades[group.iloc[i]['rfqCounterparty']].append(current_ladder)
        return ladder_trades

    ladder_trades = detect_ladder_trading(filtered_df)
    ladder_counts = {k: len(v) for k, v in ladder_trades.items()}
    top_ladder_traders = sorted(ladder_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig = px.bar(x=[trader[0] for trader in top_ladder_traders], y=[trader[1] for trader in top_ladder_traders],
                 labels={'x': 'Counterparty', 'y': 'Number of Potential Ladder Trades'},
                 title='Top 10 Potential Ladder Traders')
    st.plotly_chart(fig)

    # Example of a specific ladder trade
    if ladder_trades:
        example_counterparty = list(ladder_trades.keys())[0]
        example_ladder = ladder_trades[example_counterparty][0]
        st.write(f"Example of a potential ladder trade by {example_counterparty}:")
        st.write(pd.DataFrame([trade[['datetime', 'sym', 'rfqL0DealQty']] for trade in example_ladder]))

with tab4:
    st.header('Liquidity Analysis')

    # Overall Liquidity Score Distribution
    st.subheader('Overall Liquidity Score Distribution')
    fig = px.histogram(df, x='liquidityScore', nbins=50, labels={'liquidityScore': 'Liquidity Score'},
                       title='Distribution of Liquidity Scores')
    st.plotly_chart(fig)

    # Liquidity Score vs Deal Size
    st.subheader('Liquidity Score vs Deal Size')
    fig = px.scatter(df, x='liquidityScore', y='rfqL0DealQty', color='rfqL0VerbStrid',
                     labels={'liquidityScore': 'Liquidity Score', 'rfqL0DealQty': 'Deal Quantity'},
                     title='Liquidity Score vs Deal Size')
    st.plotly_chart(fig)

    # Average Liquidity Score by Security
    avg_liquidity_by_security = df.groupby('sym')['liquidityScore'].mean().sort_values(ascending=False)
    st.subheader('Top 10 Securities by Average Liquidity Score')
    fig = px.bar(x=avg_liquidity_by_security.head(10).index, y=avg_liquidity_by_security.head(10).values,
                 labels={'x': 'Security', 'y': 'Average Liquidity Score'})
    st.plotly_chart(fig)

    # Liquidity Score Trends Over Time
    st.subheader('Liquidity Score Trends Over Time')
    daily_avg_liquidity = df.resample('D', on='datetime')['liquidityScore'].mean()
    fig = px.line(x=daily_avg_liquidity.index, y=daily_avg_liquidity.values,
                  labels={'x': 'Date', 'y': 'Average Liquidity Score'})
    st.plotly_chart(fig)

with tab5:
    st.header('Time-based Patterns')

    # Hourly Distribution of RFQs
    st.subheader('Hourly Distribution of RFQs')
    df['hour'] = df['datetime'].dt.hour
    hourly_distribution = df['hour'].value_counts().sort_index()
    fig = px.bar(x=hourly_distribution.index, y=hourly_distribution.values,
                 labels={'x': 'Hour of Day', 'y': 'Number of RFQs'},
                 title='Hourly Distribution of RFQs')
    st.plotly_chart(fig)

    # Day of Week Analysis
    st.subheader('Day of Week Analysis')
    df['day_of_week'] = df['datetime'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_distribution = df['day_of_week'].value_counts().reindex(day_order)
    fig = px.bar(x=dow_distribution.index, y=dow_distribution.values,
                 labels={'x': 'Day of Week', 'y': 'Number of RFQs'},
                 title='RFQ Distribution by Day of Week')
    st.plotly_chart(fig)

    # Monthly Seasonality
    st.subheader('Monthly Seasonality')
    df['month'] = df['datetime'].dt.month
    monthly_distribution = df['month'].value_counts().sort_index()
    fig = px.line(x=monthly_distribution.index, y=monthly_distribution.values,
                  labels={'x': 'Month', 'y': 'Number of RFQs'},
                  title='Monthly Distribution of RFQs')
    st.plotly_chart(fig)

    # Time Between RFQs Analysis
    st.subheader('Time Between RFQs Analysis')
    df_sorted = df.sort_values('datetime')
    df_sorted['time_since_last_rfq'] = df_sorted['datetime'].diff().dt.total_seconds() / 60  # in minutes
    fig = px.histogram(df_sorted['time_since_last_rfq'].dropna(), nbins=50,
                       labels={'value': 'Minutes Since Last RFQ', 'count': 'Frequency'},
                       title='Distribution of Time Between RFQs')
    st.plotly_chart(fig)

# Add any additional analyses or visualizations here
