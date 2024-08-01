import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

def preprocess_data(df):
    df['ticker'] = df['sym'].str.split().str[0]
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    return df

def engineer_features(df):
    features = df.groupby('rfqCounterparty').agg({
        'rfqL0DealQty': ['mean', 'std', 'sum'],
        'liquidityScore': ['mean', 'std'],
        'ticker': lambda x: x.nunique(),
        'sym': lambda x: x.nunique(),
        'hour': ['mean', 'std'],
        'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
        'rfqCounterparty': 'count',
        'normalizedState': lambda x: (x == 'DONE').mean()  # Proportion of completed trades
    })
    
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features.rename(columns={'rfqCounterparty_count': 'total_trades', 'normalizedState_<lambda>': 'trade_completion_rate'}, inplace=True)
    
    date_range = (df['datetime'].max() - df['datetime'].min()).days + 1
    features['trade_frequency'] = features['total_trades'] / date_range
    features['unique_sym_percentage'] = features['sym_<lambda_0>'] / features['total_trades'] * 100
    
    time_between_trades = df.sort_values('datetime').groupby('rfqCounterparty')['datetime'].diff().dt.total_seconds() / 3600
    features['avg_time_between_trades'] = time_between_trades.groupby(df['rfqCounterparty']).mean()
    
    # Feature for trading similar bonds
    def similar_bond_ratio(group):
        total_trades = len(group)
        similar_trades = sum(group['ticker'].shift() == group['ticker'])
        return similar_trades / total_trades if total_trades > 0 else 0

    features['similar_bond_ratio'] = df.sort_values('datetime').groupby('rfqCounterparty').apply(similar_bond_ratio)
    
    return features

def run_clustering(features, n_clusters=3):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    return cluster_labels, kmeans, scaler

def run_pca(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA()
    pca_result = pca.fit_transform(scaled_features)
    
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= 0.8) + 1
    
    return pca_result[:, :n_components], pca, n_components

def plot_clusters(pca_result, labels, feature_names):
    df_plot = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    df_plot['Cluster'] = labels
    
    fig = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster', 
                     hover_data=[feature_names.index])
    fig.update_layout(title='Counterparty Clusters (PCA Visualization)')
    
    return fig

def plot_feature_importance(features, kmeans, scaler):
    feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
    feature_importance = feature_importance * scaler.scale_  # Rescale importance
    feature_names = features.columns
    
    fig = px.bar(x=feature_names, y=feature_importance, 
                 labels={'x': 'Features', 'y': 'Importance'},
                 title='Feature Importance in Clustering')
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def plot_elbow_curve(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
    
    fig = go.Figure(data=go.Scatter(x=K, y=inertias, mode='lines+markers'))
    fig.update_layout(title='Elbow Curve for K-means Clustering',
                      xaxis_title='Number of Clusters (k)',
                      yaxis_title='Inertia')
    return fig

def clustering_dashboard(df):
    st.title("Counterparty Clustering Analysis")
    
    st.header("Methodology")
    st.write("""
    1. Data Preprocessing: Extract ticker from symbol, add time-related features.
    2. Feature Engineering: Create features for each counterparty including trade statistics, 
       liquidity scores, trading patterns, and a measure for trading similar bonds.
    3. Principal Component Analysis (PCA): Reduce dimensionality while preserving variance.
    4. K-means Clustering: Group counterparties based on their trading behaviors.
    5. Visualization and Interpretation: Analyze cluster characteristics to identify VWAPers, 
       spammers, and innocent traders.
    """)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Engineer features
    features = engineer_features(df_processed)
    
    # PCA
    st.header("Principal Component Analysis")
    pca_result, pca, n_components = run_pca(features)
    st.write(f"Number of components explaining 80% of variance: {n_components}")
    
    explained_variance_ratio = pca.explained_variance_ratio_[:n_components]
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=range(1, n_components+1), y=explained_variance_ratio, name='Individual'))
    fig.add_trace(go.Scatter(x=range(1, n_components+1), y=cumulative_variance_ratio, name='Cumulative'))
    fig.update_layout(title='Explained Variance Ratio by Principal Components',
                      xaxis_title='Principal Components',
                      yaxis_title='Explained Variance Ratio')
    st.plotly_chart(fig)
    
    # Elbow curve
    st.header("Elbow Curve for K-means Clustering")
    elbow_curve = plot_elbow_curve(features)
    st.plotly_chart(elbow_curve)
    
    # Clustering
    st.header("K-means Clustering")
    n_clusters = st.slider("Number of clusters", 2, 5, 3)
    cluster_labels, kmeans, scaler = run_clustering(features, n_clusters)
    
    # Add cluster labels to features DataFrame
    features['Cluster'] = cluster_labels
    
    # Visualize clusters
    st.subheader("Cluster Visualization")
    cluster_plot = plot_clusters(pca_result, cluster_labels, features)
    st.plotly_chart(cluster_plot)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_plot = plot_feature_importance(features, kmeans, scaler)
    st.plotly_chart(importance_plot)
    
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
        avg_completion_rate = cluster_features['trade_completion_rate'].mean()
        avg_similar_bond_ratio = cluster_features['similar_bond_ratio'].mean()
        avg_trade_frequency = cluster_features['trade_frequency'].mean()
        
        st.write(f"Cluster {i}:")
        if avg_completion_rate > 0.5 and avg_similar_bond_ratio > 0.5:
            st.write("Likely VWAPers: High trade completion rate and tendency to trade similar bonds.")
        elif avg_similar_bond_ratio > 0.5 and avg_completion_rate <= 0.5:
            st.write("Likely Spammers: Low trade completion rate but high tendency to inquire about similar bonds.")
        else:
            st.write("Likely Innocent Traders: Lower tendency to trade similar bonds repeatedly.")
        
        st.write(f"Average Trade Completion Rate: {avg_completion_rate:.2f}")
        st.write(f"Average Similar Bond Ratio: {avg_similar_bond_ratio:.2f}")
        st.write(f"Average Trade Frequency: {avg_trade_frequency:.2f} trades per day")
        st.write("---")
    
    # Top counterparties in each cluster
    st.subheader("Top Counterparties in Each Cluster")
    for i in range(n_clusters):
        st.write(f"Cluster {i} Top 10 Counterparties:")
        cluster_counterparties = features[features['Cluster'] == i].sort_values('total_trades', ascending=False).head(10)
        st.write(cluster_counterparties[['total_trades', 'trade_frequency', 'unique_sym_percentage', 'trade_completion_rate', 'similar_bond_ratio']])
        st.write("---")

# In your Streamlit app
clustering_dashboard(df)
