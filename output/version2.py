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

st.title('Focused Counterparty Analysis Dashboard')

# Main filters
st.sidebar.header('Filters')
time_diff = st.sidebar.slider('Time Difference for Repeated Trades (hours)', 1, 168, 24)
num_days = st.sidebar.slider('Number of Days to Analyze', 1, 30, 7)
min_trades = st.sidebar.number_input('Minimum Number of Trades', 2, 100, 5)

end_date = df['datetime'].max()
start_date = end_date - timedelta(days=num_days)
filtered_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

def find_repeated_trades(df, time_threshold):
    df_sorted = df.sort_values(['rfqCounterparty', 'datetime'])
    df_sorted['time_diff'] = df_sorted.groupby('rfqCounterparty')['datetime'].diff()
    repeated = df_sorted[df_sorted['time_diff'] <= timedelta(hours=time_threshold)]
    return repeated

repeated_trades = find_repeated_trades(filtered_df, time_diff)
counterparty_trade_counts = repeated_trades['rfqCounterparty'].value_counts()
qualifying_counterparties = counterparty_trade_counts[counterparty_trade_counts >= min_trades].index

st.write(f"Number of qualifying counterparties: {len(qualifying_counterparties)}")

if len(qualifying_counterparties) == 0:
    st.write("No counterparties meet the current criteria. Please adjust the filters.")
else:
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Counterparty Overview", "Detailed Analysis", "Trading Patterns"])

    with tab1:
        st.header('Counterparty Overview')

        # Tier distribution of qualifying counterparties
        tier_counts = repeated_trades[repeated_trades['rfqCounterparty'].isin(qualifying_counterparties)]['rfqCustTierInfo'].value_counts()
        fig = px.pie(values=tier_counts.values, names=tier_counts.index, title='Customer Tiers of Qualifying Counterparties')
        st.plotly_chart(fig)

        # Trade frequency
        trade_frequency = counterparty_trade_counts[qualifying_counterparties].sort_values(ascending=False)
        fig = px.bar(x=trade_frequency.index, y=trade_frequency.values, 
                     labels={'x': 'Counterparty', 'y': 'Number of Repeated Trades'},
                     title='Trade Frequency of Qualifying Counterparties')
        st.plotly_chart(fig)

        # Average liquidity score
        avg_liquidity = repeated_trades[repeated_trades['rfqCounterparty'].isin(qualifying_counterparties)].groupby('rfqCounterparty')['liquidityScore'].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_liquidity.index, y=avg_liquidity.values, 
                     labels={'x': 'Counterparty', 'y': 'Average Liquidity Score'},
                     title='Average Liquidity Score of Qualifying Counterparties')
        st.plotly_chart(fig)

    with tab2:
        st.header('Detailed Counterparty Analysis')

        selected_counterparty = st.selectbox('Select a counterparty for detailed analysis:', qualifying_counterparties)
        
        counterparty_trades = repeated_trades[repeated_trades['rfqCounterparty'] == selected_counterparty]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Repeated Trades", len(counterparty_trades))
        with col2:
            st.metric("Customer Tier", counterparty_trades['rfqCustTierInfo'].iloc[0])
        with col3:
            st.metric("Avg Liquidity Score", f"{counterparty_trades['liquidityScore'].mean():.2f}")
        
        # Bond Preference
        bond_preference = counterparty_trades['sym'].value_counts()
        fig = px.bar(x=bond_preference.index, y=bond_preference.values, 
                     labels={'x': 'Bond', 'y': 'Number of Repeated Trades'},
                     title=f'Bond Preference for {selected_counterparty}')
        st.plotly_chart(fig)
        
        # Trade Direction
        direction_counts = counterparty_trades['rfqL0VerbStrid'].value_counts()
        fig = px.pie(values=direction_counts.values, names=direction_counts.index, 
                     title=f'Trade Direction for {selected_counterparty}')
        st.plotly_chart(fig)
        
        # Trade Size Distribution
        fig = px.histogram(counterparty_trades, x='rfqL0DealQty', nbins=20, 
                           labels={'rfqL0DealQty': 'Deal Quantity'},
                           title=f'Trade Size Distribution for {selected_counterparty}')
        st.plotly_chart(fig)

        # Time of Day Trading Pattern
        counterparty_trades['hour'] = counterparty_trades['datetime'].dt.hour
        hourly_pattern = counterparty_trades['hour'].value_counts().sort_index()
        fig = px.bar(x=hourly_pattern.index, y=hourly_pattern.values, 
                     labels={'x': 'Hour of Day', 'y': 'Number of Trades'},
                     title=f'Hourly Trading Pattern for {selected_counterparty}')
        st.plotly_chart(fig)

    with tab3:
        st.header('Trading Patterns Analysis')

        # Pairs Trading Detection
        st.subheader('Potential Pairs Trading')
        def detect_pairs_trading(df, time_threshold=timedelta(minutes=30)):
            pairs = defaultdict(int)
            for _, group in df.groupby('rfqCounterparty'):
                group = group.sort_values('datetime')
                for i in range(len(group) - 1):
                    if group.iloc[i+1]['datetime'] - group.iloc[i]['datetime'] <= time_threshold:
                        pair = tuple(sorted([group.iloc[i]['sym'], group.iloc[i+1]['sym']]))
                        pairs[pair] += 1
            return pairs

        qualifying_trades = repeated_trades[repeated_trades['rfqCounterparty'].isin(qualifying_counterparties)]
        potential_pairs = detect_pairs_trading(qualifying_trades)
        top_pairs = sorted(potential_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        pair_names = [f"{pair[0][0]} - {pair[0][1]}" for pair in top_pairs]
        pair_counts = [pair[1] for pair in top_pairs]
        
        fig = px.bar(x=pair_names, y=pair_counts, labels={'x': 'Security Pair', 'y': 'Frequency'},
                     title='Top 10 Potential Pairs Trading Combinations')
        st.plotly_chart(fig)

        # Ladder Trading Analysis
        st.subheader('Potential Ladder Trading')
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

        ladder_trades = detect_ladder_trading(qualifying_trades)
        ladder_counts = {k: len(v) for k, v in ladder_trades.items()}
        top_ladder_traders = sorted(ladder_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig = px.bar(x=[trader[0] for trader in top_ladder_traders], 
                     y=[trader[1] for trader in top_ladder_traders],
                     labels={'x': 'Counterparty', 'y': 'Number of Potential Ladder Trades'},
                     title='Top 10 Potential Ladder Traders')
        st.plotly_chart(fig)

        # Example of a specific ladder trade
        if ladder_trades:
            example_counterparty = list(ladder_trades.keys())[0]
            example_ladder = ladder_trades[example_counterparty][0]
            st.write(f"Example of a potential ladder trade by {example_counterparty}:")
            st.write(pd.DataFrame([trade[['datetime', 'sym', 'rfqL0DealQty']] for trade in example_ladder]))

        # Liquidity Seeking Behavior
        st.subheader('Liquidity Seeking Behavior')
        avg_liquidity_score = qualifying_trades.groupby('rfqCounterparty')['liquidityScore'].mean().sort_values(ascending=False)
        fig = px.bar(x=avg_liquidity_score.index, y=avg_liquidity_score.values,
                     labels={'x': 'Counterparty', 'y': 'Average Liquidity Score'},
                     title='Average Liquidity Score by Counterparty')
        st.plotly_chart(fig)

        # Time-based Patterns
        st.subheader('Time-based Trading Patterns')
        qualifying_trades['hour'] = qualifying_trades['datetime'].dt.hour
        hourly_pattern = qualifying_trades['hour'].value_counts().sort_index()
        fig = px.line(x=hourly_pattern.index, y=hourly_pattern.values,
                      labels={'x': 'Hour of Day', 'y': 'Number of Trades'},
                      title='Hourly Trading Pattern for Qualifying Counterparties')
        st.plotly_chart(fig)

# Add any additional analyses or visualizations here
