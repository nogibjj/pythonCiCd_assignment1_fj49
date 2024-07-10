import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Carry Trade Analysis Dashboard", layout="wide")

# Load the data
@st.cache_data
def load_data():
    trades = pd.read_csv('client_trades.csv')
    interest_rates = pd.read_csv('interest_rates.csv')
    return trades, interest_rates

trades, interest_rates = load_data()

# Remove NDF trades
trades = trades[~trades['sym'].str.contains('NDF', na=False)]

# Preprocess the data
trades['time'] = pd.to_datetime(trades['tradeTimestamp'])
trades['month'] = trades['time'].dt.month
trades['date'] = trades['time'].dt.date

# Merge trades with interest rates
trades = pd.merge(trades, interest_rates, left_on=['base', 'month'], right_on=['Currency', 'month'], how='left')
trades = pd.merge(trades, interest_rates, left_on=['quote', 'month'], right_on=['Currency', 'month'], how='left', suffixes=('_base', '_quote'))

# Calculate interest spread and cost to hedge
trades['interest_spread'] = abs(trades['value_base'] - trades['value_quote'])
trades['cost_to_hedge'] = trades['reval15'] - trades['reval00']
trades['normalized_cost'] = (trades['cost_to_hedge'] / trades['gbpNotional']) * 10000  # in basis points
trades['cost_per_volume'] = trades['cost_to_hedge'] / trades['size']

# Function to determine if a trade is a carry trade and its direction
def classify_carry_trade(row, threshold):
    if row['interest_spread'] >= threshold:
        if (row['value_base'] < row['value_quote'] and row['clientSide'] == 'Sell') or \
           (row['value_base'] > row['value_quote'] and row['clientSide'] == 'Buy'):
            return 'Entry'
        else:
            return 'Exit'
    return 'Not Carry'

# Streamlit app
st.title('Carry Trade Analysis Dashboard')

# Sidebar for user input
threshold = st.sidebar.slider('Interest Rate Differential Threshold (%)', 0.0, 10.0, 5.75, 0.25) / 100

# Apply carry trade classification
trades['carry_trade_type'] = trades.apply(lambda row: classify_carry_trade(row, threshold), axis=1)

# Filter for carry trades
carry_trades = trades[trades['carry_trade_type'] != 'Not Carry']

# Overview
st.header('Carry Trade Overview')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Trades", f"{len(trades):,}")
with col2:
    st.metric("Carry Trades", f"{len(carry_trades):,}")
with col3:
    st.metric("Carry Trade Percentage", f"{len(carry_trades)/len(trades)*100:.2f}%")

# Distribution of carry trades
st.subheader('Distribution of Carry Trades')
fig = px.bar(carry_trades['carry_trade_type'].value_counts().reset_index(), 
             x='index', y='carry_trade_type', 
             labels={'index': 'Trade Type', 'carry_trade_type': 'Count'},
             title='Entry vs Exit Carry Trades')
st.plotly_chart(fig)

# Average cost to hedge
st.subheader('Average Cost to Hedge')
avg_cost = carry_trades.groupby('carry_trade_type')['normalized_cost'].mean().reset_index()
fig = px.bar(avg_cost, x='carry_trade_type', y='normalized_cost',
             labels={'carry_trade_type': 'Trade Type', 'normalized_cost': 'Average Normalized Cost (bps)'},
             title='Average Cost to Hedge by Trade Type')
st.plotly_chart(fig)

# Top currency pairs for carry trades
st.subheader('Top Currency Pairs for Carry Trades')
top_pairs = carry_trades['sym'].value_counts().head(10).reset_index()
fig = px.bar(top_pairs, x='sym', y='count',
             labels={'sym': 'Currency Pair', 'count': 'Number of Trades'},
             title='Top 10 Currency Pairs for Carry Trades')
st.plotly_chart(fig)

# Scatter plot of trade size vs cost to hedge
st.subheader('Trade Size vs Cost to Hedge')
# Remove outliers (trades with size > 99th percentile)
size_threshold = carry_trades['size'].quantile(0.99)
filtered_trades = carry_trades[carry_trades['size'] <= size_threshold]

fig = px.scatter(filtered_trades, x='size', y='normalized_cost', color='carry_trade_type',
                 labels={'size': 'Trade Size', 'normalized_cost': 'Normalized Cost to Hedge (bps)'},
                 title='Trade Size vs Cost to Hedge (Excluding Outliers)',
                 hover_data=['sym', 'price', 'clientSide'])
st.plotly_chart(fig)

# Display outliers in a table
st.subheader('Outlier Trades (Size > 99th Percentile)')
outliers = carry_trades[carry_trades['size'] > size_threshold]
st.dataframe(outliers[['time', 'sym', 'size', 'normalized_cost', 'carry_trade_type']])

# Time series of carry trade activity
st.subheader('Carry Trade Activity Over Time')
daily_activity = carry_trades.resample('D', on='time').size().reset_index()
fig = px.line(daily_activity, x='time', y=0,
              labels={'time': 'Date', '0': 'Number of Trades'},
              title='Daily Carry Trade Activity')
st.plotly_chart(fig)

# Profitability analysis
st.subheader('Profitability Analysis')
carry_trades['profit'] = np.where(carry_trades['clientSide'] == 'Buy',
                                  carry_trades['price'] - carry_trades['reval15'],
                                  carry_trades['reval15'] - carry_trades['price'])
carry_trades['profit_percentage'] = carry_trades['profit'] / carry_trades['price'] * 100

# Profit over time
fig = px.scatter(carry_trades, x='time', y='profit_percentage', color='carry_trade_type',
                 labels={'time': 'Date', 'profit_percentage': 'Profit (%)', 'carry_trade_type': 'Trade Type'},
                 title='Profit Percentage Over Time')
st.plotly_chart(fig)

# Average profit by trade type
avg_profit = carry_trades.groupby('carry_trade_type')['profit_percentage'].mean().reset_index()
fig = px.bar(avg_profit, x='carry_trade_type', y='profit_percentage',
             labels={'carry_trade_type': 'Trade Type', 'profit_percentage': 'Average Profit (%)'},
             title='Average Profit by Trade Type')
st.plotly_chart(fig)

# Currency pair vs avg cost to hedge
st.subheader('Currency Pair vs Average Cost to Hedge')
pair_avg_cost = carry_trades.groupby('sym')['normalized_cost'].mean().sort_values(ascending=False).head(20).reset_index()
fig = px.bar(pair_avg_cost, x='sym', y='normalized_cost',
             labels={'sym': 'Currency Pair', 'normalized_cost': 'Average Cost to Hedge (bps)'},
             title='Top 20 Currency Pairs by Average Cost to Hedge')
st.plotly_chart(fig)

# Client ID vs avg cost to hedge
st.subheader('Client ID vs Average Cost to Hedge')
client_avg_cost = carry_trades.groupby('groupId')['normalized_cost'].mean().sort_values(ascending=False).head(20).reset_index()
fig = px.bar(client_avg_cost, x='groupId', y='normalized_cost',
             labels={'groupId': 'Client ID', 'normalized_cost': 'Average Cost to Hedge (bps)'},
             title='Top 20 Clients by Average Cost to Hedge')
st.plotly_chart(fig)

# Cost per volume analysis
st.subheader('Cost per Volume Analysis')
carry_trades['cost_per_volume_bps'] = carry_trades['cost_per_volume'] * 10000  # Convert to basis points
avg_cost_per_volume = carry_trades.groupby('carry_trade_type')['cost_per_volume_bps'].mean().reset_index()
fig = px.bar(avg_cost_per_volume, x='carry_trade_type', y='cost_per_volume_bps',
             labels={'carry_trade_type': 'Trade Type', 'cost_per_volume_bps': 'Average Cost per Volume (bps)'},
             title='Average Cost per Volume by Trade Type')
st.plotly_chart(fig)

# Currency Pair Time Series Analysis
st.header('Currency Pair Time Series Analysis')

# Get unique currency pairs
currency_pairs = carry_trades['sym'].unique()

# Allow user to select a currency pair
selected_pair = st.selectbox('Select a currency pair', currency_pairs)

# Filter data for the selected pair
pair_data = trades[trades['sym'] == selected_pair]
pair_carry_trades = carry_trades[carry_trades['sym'] == selected_pair]

# Create the time series plot
fig = go.Figure()

# Add the price line
fig.add_trace(go.Scatter(
    x=pair_data['time'],
    y=pair_data['price'],
    mode='lines',
    name='Price'
))

# Add markers for carry trades
for trade_type in ['Entry', 'Exit']:
    type_data = pair_carry_trades[pair_carry_trades['carry_trade_type'] == trade_type]
    fig.add_trace(go.Scatter(
        x=type_data['time'],
        y=type_data['price'],
        mode='markers',
        name=f'Carry Trade {trade_type}',
        marker=dict(
            size=8,
            symbol='circle',
            color='red' if trade_type == 'Exit' else 'green'
        ),
        text=[f"Type: {row['carry_trade_type']}<br>Client ID: {row['groupId']}<br>Size: {row['size']:.2f}<br>GBP Notional: {row['gbpNotional']:.2f}<br>Cost to Hedge (bps): {row['normalized_cost']:.2f}" for _, row in type_data.iterrows()],
        hoverinfo='text'
    ))

# Update layout
fig.update_layout(
    title=f'{selected_pair} Price and Carry Trades',
    xaxis_title='Date',
    yaxis_title='Price',
    hovermode='closest'
)

# Show the plot
st.plotly_chart(fig, use_container_width=True)

# Summary statistics for the selected pair
st.subheader(f'Summary Statistics for {selected_pair}')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Trades", f"{len(pair_data):,}")
with col2:
    st.metric("Carry Trades", f"{len(pair_carry_trades):,}")
with col3:
    st.metric("Carry Trade Percentage", f"{len(pair_carry_trades)/len(pair_data)*100:.2f}%")

# Carry trade details for the selected pair
st.subheader(f'Carry Trade Details for {selected_pair}')
st.dataframe(pair_carry_trades[['time', 'price', 'clientSide', 'size', 'gbpNotional', 'carry_trade_type', 'groupId', 'normalized_cost', 'profit_percentage']])

# Footer
st.markdown("---")
st.write("Note: This dashboard provides an analysis of carry trades based on the provided data. "
         "Always consult with financial experts for investment decisions.")
