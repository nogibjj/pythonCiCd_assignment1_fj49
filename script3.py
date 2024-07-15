## table
# GBP Notional Range Analysis
st.header('GBP Notional Range Analysis')

gbp_notional_analysis = carry_trades.groupby('gbp_notional_range').agg({
    'gbpNotional': 'count',
    'normalized_cost': 'mean'
}).reset_index()

gbp_notional_analysis.columns = ['GBP Notional Range', 'Number of Trades', 'Avg Cost to Hedge (bps)']
gbp_notional_analysis = gbp_notional_analysis.sort_values('GBP Notional Range', key=lambda x: x.map({'< 1M': 0, '1M - 2M': 1, '2M - 5M': 2, '5M - 10M': 3, '10M - 20M': 4, '> 20M': 5}))

# Display the table
st.table(gbp_notional_analysis)

# Create a bar chart
fig = px.bar(gbp_notional_analysis, x='GBP Notional Range', y='Avg Cost to Hedge (bps)',
             text='Number of Trades', 
             labels={'Avg Cost to Hedge (bps)': 'Average Cost to Hedge (bps)'},
             title='Average Cost to Hedge by GBP Notional Range')
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
st.plotly_chart(fig)

# Scatter plot
fig = px.scatter(carry_trades, x='gbpNotional', y='normalized_cost', color='gbp_notional_range',
                 labels={'gbpNotional': 'GBP Notional', 'normalized_cost': 'Cost to Hedge (bps)'},
                 title='Cost to Hedge vs GBP Notional',
                 hover_data=['sym', 'clientSide', 'carry_trade_type'])
fig.update_xaxes(type="log")
st.plotly_chart(fig)


## comparision 

st.header('Cumulative Average Cost to Hedge by Trade Size: Carry Trades vs All Trades')

# Dropdown for currency types
currency_group = st.selectbox('Select Currency Group', ['All', 'G10', 'G7'])

def prepare_data(df):
    # Sort trades by GBP notional
    sorted_df = df.sort_values('gbpNotional')
    # Calculate cumulative average cost to hedge
    sorted_df['cumulative_avg_cost'] = sorted_df['normalized_cost'].cumsum() / (sorted_df.index + 1)
    return sorted_df

# Filter and prepare data for all trades
all_trades = trades[trades['sym'].apply(lambda x: get_currency_group(x, currency_group))]
all_trades_prepared = prepare_data(all_trades)

# Filter and prepare data for carry trades
carry_trades_filtered = carry_trades[carry_trades['sym'].apply(lambda x: get_currency_group(x, currency_group))]
carry_trades_prepared = prepare_data(carry_trades_filtered)

# Create the line plot
fig = go.Figure()

# Add line for all trades
fig.add_trace(go.Scatter(
    x=all_trades_prepared['gbpNotional'],
    y=all_trades_prepared['cumulative_avg_cost'],
    mode='lines',
    name='All Trades'
))

# Add line for carry trades
fig.add_trace(go.Scatter(
    x=carry_trades_prepared['gbpNotional'],
    y=carry_trades_prepared['cumulative_avg_cost'],
    mode='lines',
    name='Carry Trades'
))

fig.update_layout(
    title=f'Cumulative Average Cost to Hedge vs Trade Size ({currency_group} Currencies)',
    xaxis_title='Trade Size (GBP Notional)',
    yaxis_title='Cumulative Avg Cost to Hedge (bps)',
    xaxis_type="log",
    hovermode="x unified"
)

st.plotly_chart(fig)

# Display summary statistics
st.subheader('Summary Statistics')
def get_summary(df):
    return df.agg({
        'gbpNotional': ['count', 'mean', 'median', 'min', 'max'],
        'normalized_cost': ['mean', 'median', 'min', 'max']
    }).round(2)

all_summary = get_summary(all_trades)
carry_summary = get_summary(carry_trades_filtered)

summary = pd.concat([all_summary, carry_summary], axis=1, keys=['All Trades', 'Carry Trades'])
summary.columns = ['All Trades', 'Carry Trades']
summary.index = ['Trade Count', 'Avg Trade Size', 'Median Trade Size', 'Min Trade Size', 'Max Trade Size',
                 'Avg Cost to Hedge', 'Median Cost to Hedge', 'Min Cost to Hedge', 'Max Cost to Hedge']
st.table(summary)
