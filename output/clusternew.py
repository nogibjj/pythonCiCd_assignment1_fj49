import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def preprocess_data(df):
    df['ticker'] = df['sym'].str.split().str[0]
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    return df

def engineer_features(df, similarity_type):
    features = df.groupby('rfqCounterparty').agg({
        'rfqL0DealQty': ['mean', 'std', 'sum'],
        'liquidityScore': ['mean', 'std'],
        'ticker': lambda x: x.nunique(),
        'sym': lambda x: x.nunique(),
        'hour': ['mean', 'std'],
        'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
        'rfqCounterparty': 'count',
        'normalizedState': lambda x: (x == 'DONE').mean()
    })
    
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features.rename(columns={'rfqCounterparty_count': 'total_trades', 'normalizedState_<lambda>': 'trade_completion_rate'}, inplace=True)
    
    date_range = (df['datetime'].max() - df['datetime'].min()).days + 1
    features['trade_frequency'] = features['total_trades'] / date_range
    features['unique_sym_percentage'] = features['sym_<lambda_0>'] / features['total_trades'] * 100
    
    time_between_trades = df.sort_values('datetime').groupby('rfqCounterparty')['datetime'].diff().dt.total_seconds() / 3600
    features['avg_time_between_trades'] = time_between_trades.groupby(df['rfqCounterparty']).mean()
    
    def similar_bond_ratio(group):
        total_trades = len(group)
        similar_trades = sum(group[similarity_type].shift() == group[similarity_type])
        return similar_trades / total_trades if total_trades > 0 else 0

    features['similar_bond_ratio'] = df.sort_values('datetime').groupby('rfqCounterparty').apply(similar_bond_ratio)
    
    return features

def run_clustering(features, n_clusters=3):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    return cluster_labels, kmeans, scaler

def plot_feature_importance(features, kmeans, scaler):
    feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
    feature_importance = feature_importance * scaler.scale_
    feature_names = features.columns
    
    fig = px.bar(x=feature_names, y=feature_importance, 
                 labels={'x': 'Features', 'y': 'Importance'},
                 title='Feature Importance in Clustering')
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def plot_parallel_coordinates(features, cluster_labels):
    features_normalized = (features - features.min()) / (features.max() - features.min())
    features_normalized['Cluster'] = cluster_labels
    
    fig = px.parallel_coordinates(features_normalized, color="Cluster", 
                                  labels={col: col for col in features.columns},
                                  title="Parallel Coordinates Plot of Clusters")
    return fig

def plot_radar_chart(features, cluster_labels):
    avg_features = features.groupby(cluster_labels).mean()
    
    fig = make_subplots(rows=1, cols=len(avg_features), 
                        subplot_titles=[f"Cluster {i}" for i in range(len(avg_features))],
                        specs=[[{"type": "polar"}] * len(avg_features)])
    
    for i, cluster_features in enumerate(avg_features.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=cluster_features[1],
            theta=cluster_features[1].index,
            fill='toself',
            name=f'Cluster {i}'
        ), 1, i+1)
    
    fig.update_layout(height=600, width=300*len(avg_features), title_text="Radar Charts of Cluster Characteristics")
    return fig

def clustering_dashboard(df):
    st.title("Advanced Counterparty Clustering Analysis")
    
    similarity_type = st.selectbox("Select similarity type for bonds:", 
                                   ['sym', 'ticker', 'industrySector', 'RatingSANDP'])
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Engineer features
    all_features = engineer_features(df_processed, similarity_type)
    
    # Feature selection
    st.subheader("Select features for clustering:")
    feature_options = all_features.columns.tolist()
    selected_features = st.multiselect("Choose features:", feature_options, default=['similar_bond_ratio'])
    
    if not selected_features:
        st.warning("Please select at least one feature for clustering.")
        return
    
    features = all_features[selected_features]
    
    # Clustering
    n_clusters = st.slider("Number of clusters", 2, 5, 3)
    cluster_labels, kmeans, scaler = run_clustering(features, n_clusters)
    
    # Add cluster labels to features DataFrame
    features['Cluster'] = cluster_labels
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_plot = plot_feature_importance(features, kmeans, scaler)
    st.plotly_chart(importance_plot)
    
    # Parallel Coordinates Plot
    st.subheader("Parallel Coordinates Plot")
    parallel_plot = plot_parallel_coordinates(features, cluster_labels)
    st.plotly_chart(parallel_plot)
    
    # Radar Chart
    st.subheader("Radar Charts of Cluster Characteristics")
    radar_plot = plot_radar_chart(features, cluster_labels)
    st.plotly_chart(radar_plot)
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    for i in range(n_clusters):
        st.write(f"Cluster {i}:")
        cluster_features = features[features['Cluster'] == i]
        st.write(cluster_features.mean())
        st.write("---")
    
    # Interpret clusters
    st.header("Cluster Interpretation")
    for i in range(n_clusters):
        cluster_features = features[features['Cluster'] == i]
        avg_completion_rate = cluster_features['trade_completion_rate'].mean() if 'trade_completion_rate' in cluster_features else 'N/A'
        avg_similar_bond_ratio = cluster_features['similar_bond_ratio'].mean() if 'similar_bond_ratio' in cluster_features else 'N/A'
        avg_trade_frequency = cluster_features['trade_frequency'].mean() if 'trade_frequency' in cluster_features else 'N/A'
        
        st.write(f"Cluster {i}:")
        if avg_completion_rate != 'N/A' and avg_similar_bond_ratio != 'N/A':
            if avg_completion_rate > 0.5 and avg_similar_bond_ratio > 0.5:
                st.write(f"Likely VWAPers: High trade completion rate and tendency to trade similar {similarity_type}s.")
            elif avg_similar_bond_ratio > 0.5 and avg_completion_rate <= 0.5:
                st.write(f"Likely Spammers: Low trade completion rate but high tendency to inquire about similar {similarity_type}s.")
            else:
                st.write(f"Likely Innocent Traders: Lower tendency to trade similar {similarity_type}s repeatedly.")
        
        st.write(f"Average Trade Completion Rate: {avg_completion_rate}")
        st.write(f"Average Similar {similarity_type.capitalize()} Ratio: {avg_similar_bond_ratio}")
        st.write(f"Average Trade Frequency: {avg_trade_frequency}")
        st.write("---")
    
    # Top counterparties in each cluster
    st.subheader("Top Counterparties in Each Cluster")
    for i in range(n_clusters):
        st.write(f"Cluster {i} Top 10 Counterparties:")
        cluster_counterparties = all_features[all_features.index.isin(features[features['Cluster'] == i].index)]
        cluster_counterparties = cluster_counterparties.sort_values('total_trades', ascending=False).head(10)
        st.write(cluster_counterparties[['total_trades', 'trade_frequency', 'unique_sym_percentage', 'trade_completion_rate', 'similar_bond_ratio']])
        st.write("---")
    
    # Additional Analysis
    st.header("Additional Analysis")
    
    # Similarity Distribution
    st.subheader(f"Distribution of Similar {similarity_type.capitalize()} Ratio")
    fig = px.histogram(features, x='similar_bond_ratio', nbins=50, color='Cluster',
                       title=f'Distribution of Similar {similarity_type.capitalize()} Ratio')
    st.plotly_chart(fig)
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    correlation_matrix = features.corr()
    fig = px.imshow(correlation_matrix, title="Feature Correlation Matrix")
    st.plotly_chart(fig)
    
    # Trade Completion Rate vs Similar Bond Ratio
    if 'trade_completion_rate' in features.columns and 'similar_bond_ratio' in features.columns:
        st.subheader(f"Trade Completion Rate vs Similar {similarity_type.capitalize()} Ratio")
        fig = px.scatter(features, x='similar_bond_ratio', y='trade_completion_rate', 
                         color='Cluster', hover_data=['total_trades'],
                         title=f'Trade Completion Rate vs Similar {similarity_type.capitalize()} Ratio')
        st.plotly_chart(fig)
    
    # Time Series Analysis
    st.subheader("Time Series Analysis of Similar Bond Trading")
    daily_similar_ratio = df.groupby(df['datetime'].dt.date).apply(
        lambda x: (x[similarity_type].shift() == x[similarity_type]).mean()
    )
    fig = px.line(x=daily_similar_ratio.index, y=daily_similar_ratio.values,
                  title=f'Daily Ratio of Similar {similarity_type.capitalize()} Trading')
    st.plotly_chart(fig)

# In your Streamlit app
clustering_dashboard(df)
