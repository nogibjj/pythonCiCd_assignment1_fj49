import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Load and prepare data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    def parse_datetime(date, time_start):
        date = pd.to_datetime(date)
        time_parts = time_start.split()
        if len(time_parts) == 3:  # Format: "0 days HH:MM:SS.fraction"
            time = pd.to_timedelta(time_parts[2])
        else:  # Format: "HH:MM:SS.fraction"
            time = pd.to_timedelta(time_parts[0])
        return date + time

    df["DateTime"] = df.apply(
        lambda row: parse_datetime(row["Date"], row["TimeStart"]), axis=1
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


rty_df = load_data("rty.csv")
spx_df = load_data("spx.csv")


def calculate_returns(df, start_time, end_time):
    mask = (df["DateTime"].dt.time >= start_time) & (df["DateTime"].dt.time < end_time)
    return df[mask].groupby("Date")["LogReturn"].sum()


# Streamlit app
st.title("RTY Futures Returns Analysis")

# Normalization toggle
normalize_returns = st.checkbox("Normalize Returns by SPX", value=True)

# Cumulative returns toggle
show_cumulative_returns = st.checkbox("Show Cumulative Returns", value=True)

# Time slot inputs
col1, col2 = st.columns(2)
with col1:
    time1_start = st.slider(
        "Time Slot 1 Start (Hour)", min_value=0, max_value=23, value=9, step=1
    )
    time1_end = st.slider(
        "Time Slot 1 End (Hour)", min_value=0, max_value=23, value=9, step=1
    )
    time1_start_min = st.slider(
        "Time Slot 1 Start (Minute)", min_value=0, max_value=59, value=0, step=1
    )
    time1_end_min = st.slider(
        "Time Slot 1 End (Minute)", min_value=0, max_value=59, value=30, step=1
    )
with col2:
    time2_start = st.slider(
        "Time Slot 2 Start (Hour)", min_value=0, max_value=23, value=9, step=1
    )
    time2_end = st.slider(
        "Time Slot 2 End (Hour)", min_value=0, max_value=23, value=10, step=1
    )
    time2_start_min = st.slider(
        "Time Slot 2 Start (Minute)", min_value=0, max_value=59, value=30, step=1
    )
    time2_end_min = st.slider(
        "Time Slot 2 End (Minute)", min_value=0, max_value=59, value=0, step=1
    )

time1_start = pd.to_datetime(
    f"{time1_start:02d}:{time1_start_min:02d}", format="%H:%M"
).time()
time1_end = pd.to_datetime(
    f"{time1_end:02d}:{time1_end_min:02d}", format="%H:%M"
).time()
time2_start = pd.to_datetime(
    f"{time2_start:02d}:{time2_start_min:02d}", format="%H:%M"
).time()
time2_end = pd.to_datetime(
    f"{time2_end:02d}:{time2_end_min:02d}", format="%H:%M"
).time()

if st.button("Update Analysis"):
    rty_returns_1 = calculate_returns(rty_df, time1_start, time1_end)
    rty_returns_2 = calculate_returns(rty_df, time2_start, time2_end)

    spx_returns_1 = calculate_returns(spx_df, time1_start, time1_end)
    spx_returns_2 = calculate_returns(spx_df, time2_start, time2_end)

    plot_df = pd.DataFrame(
        {
            "RTY_Returns_1": rty_returns_1,
            "RTY_Returns_2": rty_returns_2,
            "SPX_Returns_1": spx_returns_1,
            "SPX_Returns_2": spx_returns_2,
        }
    )

    if normalize_returns:
        plot_df["Returns_1"] = plot_df["RTY_Returns_1"] - plot_df["SPX_Returns_1"]
        plot_df["Returns_2"] = plot_df["RTY_Returns_2"] - plot_df["SPX_Returns_2"]
    else:
        plot_df["Returns_1"] = plot_df["RTY_Returns_1"]
        plot_df["Returns_2"] = plot_df["RTY_Returns_2"]

    plot_df["DaysAgo"] = (plot_df.index.max() - plot_df.index).days

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    plot_df[["Returns_1", "Returns_2"]] = imputer.fit_transform(
        plot_df[["Returns_1", "Returns_2"]]
    )

    # K-means Clustering
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(plot_df[["Returns_1", "Returns_2"]])

    num_clusters = 3  # You can change this to the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(scaled_returns)
    plot_df["Cluster"] = kmeans.labels_

    # Scatter plot
    fig1 = go.Figure()
    for i in range(num_clusters):
        cluster_df = plot_df[plot_df["Cluster"] == i]
        fig1.add_trace(
            go.Scatter(
                x=cluster_df["Returns_1"],
                y=cluster_df["Returns_2"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=cluster_df["DaysAgo"],
                    colorscale="Viridis",
                    colorbar=dict(title="Days Ago"),
                ),
                text=cluster_df.index.strftime("%Y-%m-%d"),
                hovertemplate="Date: %{text}<br>Returns "
                + time1_start.strftime("%H:%M")
                + "-"
                + time1_end.strftime("%H:%M")
                + ": %{x:.4f}<br>Returns "
                + time2_start.strftime("%H:%M")
                + "-"
                + time2_end.strftime("%H:%M")
                + ": %{y:.4f}<extra></extra>",
                name=f"Cluster {i+1}",
            )
        )

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df["Returns_1"], plot_df["Returns_2"]
    )
    line_x = np.array([plot_df["Returns_1"].min(), plot_df["Returns_1"].max()])
    line_y = slope * line_x + intercept

    fig1.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Regression Line"))

    # Add cluster centroids
    centroids = kmeans.cluster_centers_
    centroids_unscaled = scaler.inverse_transform(centroids)

    fig1.add_trace(
        go.Scatter(
            x=centroids_unscaled[:, 0],
            y=centroids_unscaled[:, 1],
            mode="markers",
            marker=dict(size=15, color="red", symbol="cross"),
            name="Centroids",
        )
    )

    returns_title = (
        "Normalized RTY Futures Returns" if normalize_returns else "RTY Futures Returns"
    )
    fig1.update_layout(
        title=f'{returns_title}: {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")} vs {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
        xaxis_title=f'Returns {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")}',
        yaxis_title=f'Returns {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
        legend=dict(x=0, y=1, orientation="h"),
    )

    st.plotly_chart(fig1)

    # Clustering plot
    fig_clusters = go.Figure()

    for i in range(num_clusters):
        cluster_df = plot_df[plot_df["Cluster"] == i]
        fig_clusters.add_trace(
            go.Scatter(
                x=cluster_df["Returns_1"],
                y=cluster_df["Returns_2"],
                mode="markers",
                marker=dict(size=10),
                text=cluster_df.index.strftime("%Y-%m-%d"),
                hovertemplate="Date: %{text}<br>Returns "
                + time1_start.strftime("%H:%M")
                + "-"
                + time1_end.strftime("%H:%M")
                + ": %{x:.4f}<br>Returns "
                + time2_start.strftime("%H:%M")
                + "-"
                + time2_end.strftime("%H:%M")
                + ": %{y:.4f}<extra></extra>",
                name=f"Cluster {i+1}",
            )
        )

    fig_clusters.add_trace(
        go.Scatter(
            x=centroids_unscaled[:, 0],
            y=centroids_unscaled[:, 1],
            mode="markers",
            marker=dict(size=15, color="red", symbol="cross"),
            name="Centroids",
        )
    )

    clustering_title = (
        "Clustering of Normalized Returns"
        if normalize_returns
        else "Clustering of Returns"
    )
    fig_clusters.update_layout(
        title=f'{clustering_title}: {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")} vs {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
        xaxis_title=f'Returns {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")}',
        yaxis_title=f'Returns {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
    )

    st.plotly_chart(fig_clusters)

    # Distribution plot
    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x=plot_df["Returns_1"],
            name=f'{time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")}',
            opacity=0.5,
        )
    )
    fig2.add_trace(
        go.Histogram(
            x=plot_df["Returns_2"],
            name=f'{time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
            opacity=0.5,
        )
    )
    distribution_title = (
        "Distribution of Normalized Returns"
        if normalize_returns
        else "Distribution of Returns"
    )
    fig2.update_layout(title=distribution_title, barmode="overlay")

    st.plotly_chart(fig2)

    # Simplified Heatmap of average returns by hour
    rty_df["Hour"] = rty_df["DateTime"].dt.hour
    spx_df["Hour"] = spx_df["DateTime"].dt.hour

    rty_heatmap_data = rty_df.groupby("Hour")["LogReturn"].mean()

    if normalize_returns:
        spx_heatmap_data = spx_df.groupby("Hour")["LogReturn"].mean()
        heatmap_data = rty_heatmap_data - spx_heatmap_data
        heatmap_title = "Average Normalized Returns by Hour"
    else:
        heatmap_data = rty_heatmap_data
        heatmap_title = "Average Returns by Hour"

    fig3 = go.Figure(
        data=go.Heatmap(
            z=[heatmap_data],
            x=heatmap_data.index,
            y=["Average Return"],
            colorscale="Viridis",
            colorbar=dict(title="Average Return"),
        )
    )

    fig3.update_layout(title=heatmap_title, xaxis_title="Hour of Day")

    st.plotly_chart(fig3)

    # Volatility Analysis
    volatility_1 = plot_df["Returns_1"].std() * np.sqrt(252)  # Annualized volatility
    volatility_2 = plot_df["Returns_2"].std() * np.sqrt(252)  # Annualized volatility

    # Volatility over time
    plot_df["RollingVol_1"] = plot_df["Returns_1"].rolling(window=20).std() * np.sqrt(
        252
    )
    plot_df["RollingVol_2"] = plot_df["Returns_2"].rolling(window=20).std() * np.sqrt(
        252
    )

    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["RollingVol_1"],
            mode="lines",
            name=f'Volatility {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")}',
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["RollingVol_2"],
            mode="lines",
            name=f'Volatility {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
        )
    )
    volatility_title = (
        "20-Day Rolling Volatility of Normalized Returns"
        if normalize_returns
        else "20-Day Rolling Volatility"
    )
    fig4.update_layout(
        title=volatility_title, xaxis_title="Date", yaxis_title="Annualized Volatility"
    )

    st.plotly_chart(fig4)

    # Calculate and display statistics
    correlation = plot_df["Returns_1"].corr(plot_df["Returns_2"])
    sharpe_ratio_1 = (
        np.sqrt(252) * plot_df["Returns_1"].mean() / plot_df["Returns_1"].std()
    )
    sharpe_ratio_2 = (
        np.sqrt(252) * plot_df["Returns_2"].mean() / plot_df["Returns_2"].std()
    )

    st.subheader("Statistics:")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Correlation: {correlation:.4f}")
        st.write(f"R-squared: {r_value**2:.4f}")
        st.write(f"Regression Slope: {slope:.4f}")
        st.write(f"Regression Intercept: {intercept:.4f}")
    with col2:
        st.write(
            f"Sharpe Ratio ({time1_start.strftime('%H:%M')}-{time1_end.strftime('%H:%M')}): {sharpe_ratio_1:.4f}"
        )
        st.write(
            f"Sharpe Ratio ({time2_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}): {sharpe_ratio_2:.4f}"
        )
        st.write(
            f"Volatility ({time1_start.strftime('%H:%M')}-{time1_end.strftime('%H:%M')}): {volatility_1:.4f}"
        )
        st.write(
            f"Volatility ({time2_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}): {volatility_2:.4f}"
        )

    st.write(
        f"The data points are grouped into {num_clusters} clusters based on their similarity in returns. Each cluster is represented by a different color in the scatter plot."
    )

    # Additional Analysis: Cumulative Returns
    if show_cumulative_returns:
        plot_df["CumReturns_1"] = (1 + plot_df["Returns_1"]).cumprod()
        plot_df["CumReturns_2"] = (1 + plot_df["Returns_2"]).cumprod()

        fig5 = go.Figure()
        fig5.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["CumReturns_1"],
                mode="lines",
                name=f'Cumulative Returns {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")}',
            )
        )
        fig5.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df["CumReturns_2"],
                mode="lines",
                name=f'Cumulative Returns {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
            )
        )

        cumulative_returns_title = (
            "Cumulative Normalized Returns Over Time"
            if normalize_returns
            else "Cumulative Returns Over Time"
        )
        fig5.update_layout(
            title=cumulative_returns_title,
            xaxis_title="Date",
            yaxis_title="Cumulative Returns",
        )

        st.plotly_chart(fig5)

        st.write(
            "The cumulative returns graph shows the growth of an initial investment of 1 unit over time, assuming the returns are reinvested. A value of 1.02 means that the investment has grown by 2% over the period."
        )
