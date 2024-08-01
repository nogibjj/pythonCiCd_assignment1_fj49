def trade_repetition_likelihood(df):
    st.header("Trade Repetition Likelihood Analysis")

    # Group by counterparty and symbol, count occurrences
    trade_counts = df.groupby(['rfqCounterparty', 'sym']).size().reset_index(name='count')
    
    # Calculate repetition likelihoods for each counterparty
    counterparties = df['rfqCounterparty'].unique()
    
    for counterparty in counterparties:
        st.subheader(f"Trade Repetition Likelihood for {counterparty}")
        
        counterparty_counts = trade_counts[trade_counts['rfqCounterparty'] == counterparty]['count']
        
        repetition_counts = counterparty_counts.value_counts().sort_index()
        repetition_counts = repetition_counts[repetition_counts.index >= 2]  # Start from 2 trades
        
        if len(repetition_counts) > 0:
            likelihoods = (repetition_counts.cumsum()[::-1] / repetition_counts.sum()) * 100
            
            # Create DataFrame for display and plotting
            likelihood_df = pd.DataFrame({
                'Trades': likelihoods.index,
                'Likelihood (%)': likelihoods.values
            })
            
            st.write(likelihood_df)
            
            fig = px.line(likelihood_df, x='Trades', y='Likelihood (%)', 
                          title=f'Likelihood of Trade Repetition for {counterparty}')
            fig.update_layout(xaxis_title='Number of Trades', yaxis_title='Likelihood (%)')
            st.plotly_chart(fig)
        else:
            st.write("No repeated trades found for this counterparty.")
        
        st.write("---")

# In your analysis section
trade_repetition_likelihood(df)


####

def vwaper_inquiry_analysis(df):
    st.header('VWAPER Inquiry Analysis')
    vwaper_df = df[df['category'] == 'VWAPER']
    
    # Add filters
    min_trades = st.slider('Minimum number of trades', 1, 100, 1)
    min_trade_rate = st.slider('Minimum trade rate (%)', 0.0, 100.0, 0.0)
    
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
    
    # Apply filters
    filtered_df = analysis_df[(analysis_df['Total Inquiries'] >= min_trades) & 
                              (analysis_df['Trade Rate (%)'] >= min_trade_rate)]
    
    st.write("VWAPER Inquiry Analysis:")
    st.write(filtered_df)

    # Visualize
    fig = px.scatter(filtered_df, x='Trade Rate (%)', y='Price Shown Rate (%)',
                     hover_name=filtered_df.index, size='Total Inquiries',
                     title='VWAPER Trade Rate vs Price Shown Rate',
                     labels={'index': 'Counterparty'})
    st.plotly_chart(fig)

# In your VWAPER tab
vwaper_inquiry_analysis(categorized_df)
