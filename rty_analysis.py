import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        padding-left: -10rem;
        padding-right: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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

volume_df = load_data("volume_RTY_2022.csv")

volume_df["NormalizedVolume"] = volume_df["volume"] / volume_df["volume"].sum()

# Merge return and volume data
merged_df = pd.merge(rty_df, volume_df, on="DateTime", how="left")
merged_df["NormalizedVolume"].fillna(0, inplace=True)


def calculate_returns(df, start_time, end_time):
    mask = (df["DateTime"].dt.time >= start_time) & (df["DateTime"].dt.time < end_time)
    return df[mask].groupby("Date")["LogReturn"].sum()


# Streamlit app
st.title("RTY Futures Returns Analysis")

############

# Comparisons for specific time intervals
intervals = ["1 min", "5 min", "10 min", "30 min", "1 hour"]
base_time = pd.to_datetime("09:30").time()
base_date = pd.to_datetime("2000-01-01")  # Dummy date for combining with base_time

for interval in intervals:
    st.subheader(f"Comparison for {interval} Interval")

    if interval == "1 min":
        delta = pd.Timedelta(minutes=1)
    elif interval == "5 min":
        delta = pd.Timedelta(minutes=5)
    elif interval == "10 min":
        delta = pd.Timedelta(minutes=10)
    elif interval == "30 min":
        delta = pd.Timedelta(minutes=30)
    else:  # '1 hour'
        delta = pd.Timedelta(hours=1)

    forward_time = (
        pd.to_datetime(base_date.date().isoformat() + " " + base_time.isoformat())
        + delta
    ).time()
    backward_time = (
        pd.to_datetime(base_date.date().isoformat() + " " + base_time.isoformat())
        - delta
    ).time()

    spy_forward_returns = calculate_returns(spx_df, base_time, forward_time)
    spy_backward_returns = calculate_returns(spx_df, backward_time, base_time)

    rty_forward_returns = calculate_returns(rty_df, base_time, forward_time)
    rty_backward_returns = calculate_returns(rty_df, backward_time, base_time)

    diff_forward_returns = rty_forward_returns - spy_forward_returns
    diff_backward_returns = rty_backward_returns - spy_backward_returns

    # Create a DataFrame for plotting
    plot_data = {
        "SPY": pd.DataFrame(
            {"Forward": spy_forward_returns, "Backward": spy_backward_returns}
        ),
        "RTY": pd.DataFrame(
            {"Forward": rty_forward_returns, "Backward": rty_backward_returns}
        ),
        "Diff": pd.DataFrame(
            {"Forward": diff_forward_returns, "Backward": diff_backward_returns}
        ),
    }

    # Add a dropdown to select the returns to display
    selected_returns = st.selectbox(
        "Select Returns", options=["SPY", "RTY", "Diff"], key=f"dropdown_{interval}"
    )

    # # Create subplots side by side
    # fig = make_subplots(rows=1, cols=2, subplot_titles=("Scatter Plot", "Line Graph"))

    # # Add trace for the scatter plot
    # fig.add_trace(
    #     go.Scatter(
    #         x=plot_data[selected_returns]["Backward"],
    #         y=plot_data[selected_returns]["Forward"],
    #         mode="markers",
    #         marker=dict(
    #             size=10,
    #             color=plot_data[selected_returns]["Forward"].index.map(
    #                 lambda x: (
    #                     x - plot_data[selected_returns]["Forward"].index.max()
    #                 ).days
    #             ),
    #             colorscale="Viridis",
    #             colorbar=dict(title="Days Ago"),
    #         ),
    #         hovertemplate="Backward Return: %{x:.4f}<br>Forward Return: %{y:.4f}<extra></extra>",
    #     ),
    #     row=1,
    #     col=1,
    # )

    # # Add trace for the line graph
    # fig.add_trace(
    #     go.Scatter(
    #         x=plot_data[selected_returns]["Forward"].index,
    #         y=plot_data[selected_returns]["Forward"],
    #         mode="lines",
    #         name="Forward Returns",
    #     ),
    #     row=1,
    #     col=2,
    # )

    # fig.add_trace(
    #     go.Scatter(
    #         x=plot_data[selected_returns]["Backward"].index,
    #         y=plot_data[selected_returns]["Backward"],
    #         mode="lines",
    #         name="Backward Returns",
    #     ),
    #     row=1,
    #     col=2,
    # )

    # fig.update_layout(
    #     title=f"{selected_returns} Returns Comparison for {interval} Interval",
    #     height=600,
    #     width=1400,
    #     bargap=0.1,
    #     hovermode="closest",
    #     legend=dict(
    #         x=0.05,
    #         y=0.95,
    #         bgcolor="rgba(255, 255, 255, 0.8)",
    #         bordercolor="rgba(0, 0, 0, 0.8)",
    #         borderwidth=1,
    #     ),
    #     margin=dict(l=-100),  # Adjust the left margin
    # )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Average Return across the time horizon",
            "Return Comparision before and market open",
        ),
        horizontal_spacing=0.05,
    )

    # Add trace for the scatter plot
    fig.add_trace(
        go.Scatter(
            x=plot_data[selected_returns]["Backward"],
            y=plot_data[selected_returns]["Forward"],
            mode="markers",
            marker=dict(
                size=10,
                color=plot_data[selected_returns]["Forward"].index.map(
                    lambda x: (
                        x - plot_data[selected_returns]["Forward"].index.max()
                    ).days
                ),
                colorscale="Viridis",
                colorbar=dict(title="Days Ago"),
            ),
            hovertemplate="Backward Return: %{x:.5f}<br>Forward Return: %{y:.5f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add trace for the line graph
    fig.add_trace(
        go.Scatter(
            x=plot_data[selected_returns]["Forward"].index,
            y=plot_data[selected_returns]["Forward"],
            mode="lines",
            name="Forward Returns",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_data[selected_returns]["Backward"].index,
            y=plot_data[selected_returns]["Backward"],
            mode="lines",
            name="Backward Returns",
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title=f"{selected_returns} Returns Comparison for {interval} Interval",
        height=600,
        width=2400,
        bargap=0.1,
        hovermode="closest",
        legend=dict(
            x=0.05,
            y=0.95,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.8)",
            borderwidth=1,
        ),
        xaxis1=dict(domain=[0.0, 0.45]),  # Adjust the domain for the scatter plot
        xaxis2=dict(domain=[0.55, 1.0]),  # Adjust the domain for the line plot
        margin=dict(l=0),
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Returns", row=1, col=1)
    fig.update_xaxes(title_text="Before Market Open", row=1, col=2)
    fig.update_yaxes(title_text="After Market Open", row=1, col=2)

    st.plotly_chart(fig)

    # Display statistics
    st.write("Statistics:")
    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"{selected_returns} Forward Avg Return: {plot_data[selected_returns]['Forward'].mean():.5f}"
        )
        st.write(
            f"{selected_returns} Forward Std Dev: {plot_data[selected_returns]['Forward'].std():.5f}"
        )

    with col2:
        st.write(
            f"{selected_returns} Backward Avg Return: {plot_data[selected_returns]['Backward'].mean():.5f}"
        )
        st.write(
            f"{selected_returns} Backward Std Dev: {plot_data[selected_returns]['Backward'].std():.5f}"
        )

    st.write("---")
    ##############################

# 30-minute interval returns table
st.subheader("30-Minute Interval Returns")

# Create time intervals
start_time = pd.to_datetime("09:00").time()
end_time = pd.to_datetime("16:30").time()
freq = "30min"

# Generate time intervals
base_date = pd.to_datetime("2000-01-01")  # Dummy date for combining with time
intervals = pd.date_range(
    start=base_date.replace(hour=start_time.hour, minute=start_time.minute),
    end=base_date.replace(hour=end_time.hour, minute=end_time.minute),
    freq=freq,
).time

# Get the number of unique dates in the data
num_days_data = rty_df["Date"].nunique()

# Add a slider to select the number of days
num_days = st.slider(
    "Select the number of days",
    min_value=1,
    max_value=num_days_data,
    value=num_days_data,
    step=1,
)

# Filter the data based on the selected number of days
start_date = rty_df["Date"].max() - pd.Timedelta(days=num_days - 1)
filtered_rty_df = rty_df[rty_df["Date"] >= start_date]
filtered_spx_df = spx_df[spx_df["Date"] >= start_date]

# Calculate returns for each interval
interval_returns = []

for i in range(len(intervals) - 1):
    start = intervals[i]
    end = intervals[i + 1]

    spy_returns = calculate_returns(filtered_spx_df, start, end)
    rty_returns = calculate_returns(filtered_rty_df, start, end)
    diff_returns = rty_returns - spy_returns

    interval_returns.append(
        {
            "Interval": f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}",
            "SPX Returns": spy_returns.mean(),
            "RTY Returns": rty_returns.mean(),
            "RTY - SPX Returns": diff_returns.mean(),
        }
    )

# Create a DataFrame from interval_returns
returns_df = pd.DataFrame(interval_returns)

# Format the returns to 5 decimal points
returns_df = returns_df.applymap(
    lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else x
)

# Display the table
st.table(returns_df)

### fix this, not working bec of some odd reason
# def highlight_max(val):
#     try:
#         if float(val) == filtered_returns_df.max().max():
#             return "background-color: yellow"
#         else:
#             return ""
#     except ValueError:
#         return ""


# styled_returns_df = filtered_returns_df.style.applymap(highlight_max)

# Display the table with conditional formatting
# st.table(filtered_returns_df)


###############################

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

    ######### VOLUME ############
    merged_df["Date"] = merged_df["Date_x"]
    returns_1 = calculate_returns(merged_df, time1_start, time1_end)
    returns_2 = calculate_returns(merged_df, time2_start, time2_end)

    plot_df = pd.DataFrame({"Returns_1": returns_1, "Returns_2": returns_2})
    plot_df["DaysAgo"] = (plot_df.index.max() - plot_df.index).days

    # Volume analysis
    # Volume and returns analysis
    # Volume and returns analysis
    # Volume and returns analysis
    volume_and_returns_data = merged_df[
        (merged_df["DateTime"].dt.time >= time1_start)
        & (merged_df["DateTime"].dt.time < time2_end)
    ]
    volume_and_returns_data = (
        volume_and_returns_data.groupby("Date")
        .agg({"volume": "sum", "LogReturn": "sum"})
        .reset_index()
    )

    fig_volume_returns = make_subplots(specs=[[{"secondary_y": True}]])

    fig_volume_returns.add_trace(
        go.Scatter(
            x=volume_and_returns_data["Date"],
            y=volume_and_returns_data["volume"],
            mode="lines",
            name="Volume",
        ),
        secondary_y=False,
    )

    fig_volume_returns.add_trace(
        go.Scatter(
            x=volume_and_returns_data["Date"],
            y=volume_and_returns_data["LogReturn"],
            mode="lines",
            name="Returns",
        ),
        secondary_y=True,
    )

    fig_volume_returns.update_layout(
        title=f"Volume and Returns: {time1_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}",
        xaxis_title="Date",
        yaxis_title="Volume",
        yaxis2_title="Returns",
    )

    st.plotly_chart(fig_volume_returns)

    ######### VOLUME ############
    merged_df["Date"] = merged_df["Date_x"]
    returns_1 = calculate_returns(merged_df, time1_start, time1_end)
    returns_2 = calculate_returns(merged_df, time2_start, time2_end)

    plot_df = pd.DataFrame({"Returns_1": returns_1, "Returns_2": returns_2})
    plot_df["DaysAgo"] = (plot_df.index.max() - plot_df.index).days

    # Volume and returns analysis
    volume_and_returns_data = merged_df[
        (merged_df["DateTime"].dt.time >= time1_start)
        & (merged_df["DateTime"].dt.time < time2_end)
    ]
    volume_and_returns_data = (
        volume_and_returns_data.groupby("Date")
        .agg({"volume": "sum", "LogReturn": "sum"})
        .reset_index()
    )

    fig_volume_returns = make_subplots(specs=[[{"secondary_y": True}]])

    fig_volume_returns.add_trace(
        go.Scatter(
            x=volume_and_returns_data["Date"],
            y=volume_and_returns_data["volume"],
            mode="lines",
            name="Volume",
        ),
        secondary_y=False,
    )

    fig_volume_returns.add_trace(
        go.Scatter(
            x=volume_and_returns_data["Date"],
            y=volume_and_returns_data["LogReturn"],
            mode="lines",
            name="Returns",
        ),
        secondary_y=True,
    )

    fig_volume_returns.update_layout(
        title=f"Volume and Returns: {time1_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}",
        xaxis_title="Date",
        yaxis_title="Volume",
        yaxis2_title="Returns",
    )

    st.plotly_chart(fig_volume_returns)

    # Scatter plot of volume vs returns
    fig_volume_returns_scatter = go.Figure()

    fig_volume_returns_scatter.add_trace(
        go.Scatter(
            x=volume_and_returns_data["volume"],
            y=volume_and_returns_data["LogReturn"],
            mode="markers",
            marker=dict(
                size=10,
                color=volume_and_returns_data["Date"].map(
                    lambda x: (x - volume_and_returns_data["Date"].max()).days
                ),
                colorscale="Viridis",
                colorbar=dict(title="Days Ago"),
            ),
            hovertemplate="Volume: %{x}<br>Returns: %{y:.4f}<extra></extra>",
        )
    )

    fig_volume_returns_scatter.update_layout(
        title=f"Volume vs Returns: {time1_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}",
        xaxis_title="Volume",
        yaxis_title="Returns",
    )

    st.plotly_chart(fig_volume_returns_scatter)

    # Moving average of volume
    volume_data = merged_df[
        (merged_df["DateTime"].dt.time >= time1_start)
        & (merged_df["DateTime"].dt.time < time2_end)
    ]
    volume_data = volume_data.groupby("Date").agg({"volume": "sum"}).reset_index()

    volume_data["VolumeMA10"] = volume_data["volume"].rolling(window=10).mean()
    volume_data["VolumeMA20"] = volume_data["volume"].rolling(window=20).mean()

    fig_volume_ma = go.Figure()

    fig_volume_ma.add_trace(
        go.Scatter(
            x=volume_data["Date"], y=volume_data["volume"], mode="lines", name="Volume"
        )
    )
    fig_volume_ma.add_trace(
        go.Scatter(
            x=volume_data["Date"],
            y=volume_data["VolumeMA10"],
            mode="lines",
            name="10-Day MA",
        )
    )
    fig_volume_ma.add_trace(
        go.Scatter(
            x=volume_data["Date"],
            y=volume_data["VolumeMA20"],
            mode="lines",
            name="20-Day MA",
        )
    )

    fig_volume_ma.update_layout(
        title=f"Volume and Moving Averages: {time1_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}",
        xaxis_title="Date",
        yaxis_title="Volume",
    )

    st.plotly_chart(fig_volume_ma)

    # Volume distribution
    fig_volume_dist = go.Figure()

    fig_volume_dist.add_trace(
        go.Histogram(x=volume_data["volume"], nbinsx=50, name="Volume Distribution")
    )

    fig_volume_dist.update_layout(
        title=f"Volume Distribution: {time1_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}",
        xaxis_title="Volume",
        yaxis_title="Frequency",
    )

    st.plotly_chart(fig_volume_dist)

    # # Volume analysis
    # volume_data = merged_df[
    #     (merged_df["DateTime"].dt.time >= time1_start)
    #     & (merged_df["DateTime"].dt.time < time2_end)
    # ]
    # volume_data = (
    #     volume_data.groupby("Date")
    #     .agg({"NormalizedVolume": "sum", "tradeCount": "sum"})
    #     .reset_index()
    # )

    # fig_volume = go.Figure()
    # fig_volume.add_trace(
    #     go.Scatter(
    #         x=volume_data["Date"],
    #         y=volume_data["NormalizedVolume"],
    #         mode="lines",
    #         name="Normalized Volume",
    #     )
    # )
    # fig_volume.add_trace(
    #     go.Scatter(
    #         x=volume_data["Date"],
    #         y=volume_data["tradeCount"],
    #         mode="lines",
    #         name="Trade Count",
    #         yaxis="y2",
    #     )
    # )
    # fig_volume.update_layout(
    #     title=f"Volume and Trade Count: {time1_start.strftime('%H:%M')}-{time2_end.strftime('%H:%M')}",
    #     xaxis_title="Date",
    #     yaxis_title="Normalized Volume",
    # )

    # fig_volume.update_yaxes(title_text="Normalized Volume", secondary_y=False)
    # fig_volume.update_yaxes(title_text="Trade Count", secondary_y=True)

    # st.plotly_chart(fig_volume)

    # Predictive model
    features = ["NormalizedVolume", "tradeCount"]
    target = "LogReturn"

    model_data = merged_df[
        (merged_df["DateTime"].dt.time >= time1_start)
        & (merged_df["DateTime"].dt.time < time2_end)
    ]
    model_data = model_data[features + [target]].dropna()

    X = model_data[features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Predictive Model")
    st.write("Features used: Normalized Volume, Trade Count")
    st.write("Target variable: Log Return")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-squared: {r2:.4f}")
