def trades_completion_analysis(df):
    # Analyze completed trades by client tier
    completed_trades = df[df['normalizedState'] == 'DONE'].groupby('rfqCustTierInfo').size()
    total_trades = df.groupby('rfqCustTierInfo').size()
    completion_rate = (completed_trades / total_trades * 100).round(2)

    # Analyze price shown
    price_shown = df[df['price'].notnull()].groupby('rfqCustTierInfo').size()
    price_shown_rate = (price_shown / total_trades * 100).round(2)

    # Combine results
    analysis_df = pd.DataFrame({
        'Total Trades': total_trades,
        'Completed Trades': completed_trades,
        'Completion Rate (%)': completion_rate,
        'Trades with Price Shown': price_shown,
        'Price Shown Rate (%)': price_shown_rate
    })

    st.write("Trade Completion and Price Shown Analysis by Client Tier:")
    st.write(analysis_df)

    # Visualize
    fig = px.bar(analysis_df, x=analysis_df.index, y=['Completion Rate (%)', 'Price Shown Rate (%)'],
                 title='Completion Rate and Price Shown Rate by Client Tier',
                 labels={'value': 'Percentage', 'variable': 'Metric', 'rfqCustTierInfo': 'Client Tier'})
    st.plotly_chart(fig)

# In your Overview tab
trades_completion_analysis(df)



def vwaper_client_tier_analysis(df):
    vwaper_df = df[df['category'] == 'VWAPER']
    tier_counts = vwaper_df['rfqCustTierInfo'].value_counts()
    
    st.write("VWAPER Trades by Client Tier:")
    st.write(tier_counts)

    fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                 title='Distribution of VWAPER Trades by Client Tier')
    st.plotly_chart(fig)

# In your VWAPER tab
vwaper_client_tier_analysis(categorized_df)

def trade_repetition_likelihood(df):
    # Group by counterparty and count trades
    trade_counts = df.groupby('rfqCounterparty').size().value_counts().sort_index()
    total_counterparties = len(df['rfqCounterparty'].unique())

    # Calculate probabilities
    probabilities = pd.Series(index=trade_counts.index, dtype=float)
    for i in probabilities.index:
        probabilities[i] = (trade_counts[trade_counts >= i].sum() / total_counterparties) * 100

    st.write("Likelihood of Trade Repetition:")
    st.write(probabilities)

    fig = px.line(x=probabilities.index, y=probabilities.values,
                  title='Likelihood of Trade Repetition',
                  labels={'x': 'Number of Trades', 'y': 'Likelihood (%)'})
    st.plotly_chart(fig)

# In your analysis section
trade_repetition_likelihood(df)

def liquidity_score_analysis(df):
    # Analyze liquidity score distribution
    fig = px.histogram(df, x='liquidityScore', nbins=50,
                       title='Distribution of Liquidity Scores',
                       labels={'liquidityScore': 'Liquidity Score', 'count': 'Frequency'})
    st.plotly_chart(fig)

    # Analyze average liquidity score by trade category
    avg_liquidity = df.groupby('category')['liquidityScore'].mean().sort_values(ascending=False)
    st.write("Average Liquidity Score by Trade Category:")
    st.write(avg_liquidity)

    fig = px.bar(x=avg_liquidity.index, y=avg_liquidity.values,
                 title='Average Liquidity Score by Trade Category',
                 labels={'x': 'Category', 'y': 'Average Liquidity Score'})
    st.plotly_chart(fig)

# In your analysis section
liquidity_score_analysis(categorized_df)

def vwaper_inquiry_analysis(df):
    vwaper_df = df[df['category'] == 'VWAPER']
    
    # Analyze how many inquiries were traded
    traded_count = vwaper_df[vwaper_df['normalizedState'] == 'DONE'].groupby('rfqCounterparty').size()
    total_count = vwaper_df.groupby('rfqCounterparty').size()
    trade_rate = (traded_count / total_count * 100).round(2)

    # Analyze how many inquiries were shown a price
    price_shown_count = vwaper_df[vwaper_df['price'].notnull()].groupby('rfqCounterparty').size()
    price_shown_rate = (price_shown_count / total_count * 100).round(2)

    analysis_df = pd.DataFrame({
        'Total Inquiries': total_count,
        'Traded': traded_count,
        'Trade Rate (%)': trade_rate,
        'Price Shown': price_shown_count,
        'Price Shown Rate (%)': price_shown_rate
    })

    st.write("VWAPER Inquiry Analysis:")
    st.write(analysis_df)

    # Visualize
    fig = px.scatter(analysis_df, x='Trade Rate (%)', y='Price Shown Rate (%)',
                     hover_name=analysis_df.index, size='Total Inquiries',
                     title='VWAPER Trade Rate vs Price Shown Rate',
                     labels={'index': 'Counterparty'})
    st.plotly_chart(fig)

# In your VWAPER tab
vwaper_inquiry_analysis(categorized_df)
