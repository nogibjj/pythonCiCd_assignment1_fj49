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
trades['normalized_cost'] = trades['cost_to_hedge'] / trades['gbpNotional']

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
             labels={'carry_trade_type': 'Trade Type', 'normalized_cost': 'Average Normalized Cost'},
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
fig = px.scatter(carry_trades, x='gbpNotional', y='normalized_cost', color='carry_trade_type',
                 labels={'gbpNotional': 'Trade Size (GBP Notional)', 'normalized_cost': 'Normalized Cost to Hedge'},
                 title='Trade Size vs Cost to Hedge',
                 hover_data=['sym', 'price', 'clientSide'])
fig.update_xaxes(type="log")
st.plotly_chart(fig)

# Time series of carry trade activity
st.subheader('Carry Trade Activity Over Time')
daily_activity = carry_trades.resample('D', on='time').size().reset_index()
fig = px.line(daily_activity, x='time', y=0,
              labels={'time': 'Date', '0': 'Number of Trades'},
              title='Daily Carry Trade Activity')
st.plotly_chart(fig)

# Profitability analysis
st.subheader('Profitability Analysis')
carry_trades['profit'] = np.where(carry_trades['carry_trade_type'] == 'Entry', 
                                  carry_trades['reval15'] - carry_trades['reval00'],
                                  carry_trades['reval00'] - carry_trades['reval15'])
profit_by_type = carry_trades.groupby('carry_trade_type')['profit'].sum().reset_index()
fig = px.bar(profit_by_type, x='carry_trade_type', y='profit',
             labels={'carry_trade_type': 'Trade Type', 'profit': 'Total Profit'},
             title='Profitability by Trade Type')
st.plotly_chart(fig)

# Currency pair performance
st.subheader('Currency Pair Performance')
pair_performance = carry_trades.groupby('sym')['profit'].mean().sort_values(ascending=False).head(10).reset_index()
fig = px.bar(pair_performance, x='sym', y='profit',
             labels={'sym': 'Currency Pair', 'profit': 'Average Profit'},
             title='Top 10 Currency Pairs by Average Profit')
st.plotly_chart(fig)

# Detailed view of carry trades
st.subheader('Detailed Carry Trade Data')
st.dataframe(carry_trades[['time', 'sym', 'price', 'clientSide', 'gbpNotional', 'interest_spread', 'cost_to_hedge', 'carry_trade_type', 'profit']])

# Distribution of interest rate spreads
st.subheader('Distribution of Interest Rate Spreads')
fig = px.histogram(carry_trades, x='interest_spread', nbins=30,
                   labels={'interest_spread': 'Interest Rate Spread'},
                   title='Distribution of Interest Rate Spreads')
fig.update_traces(marker_line_width=1, marker_line_color="white")
st.plotly_chart(fig)

# Correlation heatmap
st.subheader('Correlation Heatmap')
corr_matrix = carry_trades[['gbpNotional', 'interest_spread', 'cost_to_hedge', 'profit']].corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                labels=dict(x="Feature", y="Feature", color="Correlation"),
                x=corr_matrix.columns, y=corr_matrix.columns)
fig.update_layout(title='Correlation Heatmap of Key Metrics')
st.plotly_chart(fig)

# Time-based analysis
st.subheader('Carry Trade Profitability Over Time')
monthly_profit = carry_trades.resample('M', on='time')['profit'].mean().reset_index()
fig = px.line(monthly_profit, x='time', y='profit',
              labels={'time': 'Date', 'profit': 'Average Monthly Profit'},
              title='Monthly Average Carry Trade Profitability')
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
        text=[f"Type: {row['carry_trade_type']}<br>Client ID: {row['groupId']}<br>Size: {row['size']:.2f}<br>GBP Notional: {row['gbpNotional']:.2f}" for _, row in type_data.iterrows()],
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
st.dataframe(pair_carry_trades[['time', 'price', 'clientSide', 'size', 'gbpNotional', 'carry_trade_type', 'groupId']])

# Footer
st.markdown("---")
st.write("Note: This dashboard provides an analysis of carry trades based on the provided data. "
         "Always consult with financial experts for investment decisions.")
