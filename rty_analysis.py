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
from scipy.stats import ttest_ind

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

show_percentage_returns = st.checkbox("Show Percentage Returns", value=False)

st.subheader("Analysis Settings:")

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

    if show_percentage_returns:
        plot_df["Returns_1"] = (np.exp(plot_df["Returns_1"]) - 1) * 100
        plot_df["Returns_2"] = (np.exp(plot_df["Returns_2"]) - 1) * 100

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

    fig1.add_hline(y=0, line_dash="dash", line_color="black")
    fig1.add_vline(x=0, line_dash="dash", line_color="black")

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
    returns_unit = "%" if show_percentage_returns else ""
    fig1.update_layout(
        title=f'{returns_title}: {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")} vs {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")}',
        xaxis_title=f'Returns {time1_start.strftime("%H:%M")}-{time1_end.strftime("%H:%M")} {returns_unit}',
        yaxis_title=f'Returns {time2_start.strftime("%H:%M")}-{time2_end.strftime("%H:%M")} {returns_unit}',
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

    # Heatmap of average returns by day of week and hour
    st.subheader("Average Returns by Day of Week and Hour")

    rty_df["DayOfWeek"] = rty_df["DateTime"].dt.dayofweek
    rty_df["Hour"] = rty_df["DateTime"].dt.hour

    if normalize_returns:
        spx_df["DayOfWeek"] = spx_df["DateTime"].dt.dayofweek
        spx_df["Hour"] = spx_df["DateTime"].dt.hour

        rty_heatmap_data = (
            rty_df.groupby(["DayOfWeek", "Hour"])["LogReturn"].mean().unstack()
        )
        spx_heatmap_data = (
            spx_df.groupby(["DayOfWeek", "Hour"])["LogReturn"].mean().unstack()
        )
        heatmap_data = rty_heatmap_data - spx_heatmap_data
        heatmap_title = "Average Normalized Returns by Day of Week and Hour"
    else:
        heatmap_data = (
            rty_df.groupby(["DayOfWeek", "Hour"])["LogReturn"].mean().unstack()
        )
        heatmap_title = "Average Returns by Day of Week and Hour"

    # Convert to percentage if checkbox is selected
    if show_percentage_returns:
        heatmap_data = (np.exp(heatmap_data) - 1) * 100
        value_format = ".2f"
        colorbar_title = "Average Return (%)"
    else:
        value_format = ".4f"
        colorbar_title = "Average Return"

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            colorscale="RdBu",
            zmid=0,
            text=heatmap_data.values,
            texttemplate="%{text:" + value_format + "}",
            colorbar=dict(title=colorbar_title),
        )
    )

    fig_heatmap.update_layout(
        title=heatmap_title,
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    )

    st.plotly_chart(fig_heatmap)

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
# Add this import at the top of your script

# Add this section after the "Data Visualization" section and before the "Time Series Decomposition" section

#####
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ttest_ind

st.header("3. Conditional Returns Analysis")

# Date range selection
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime(rty_df["Date"].min()))
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime(rty_df["Date"].max()))

# Filter data based on selected date range
rty_df_filtered = rty_df[
    (rty_df["Date"].dt.date >= start_date) & (rty_df["Date"].dt.date <= end_date)
]


def calculate_returns(df, start_time, end_time):
    mask = (df["DateTime"].dt.time >= start_time) & (df["DateTime"].dt.time < end_time)
    return df[mask].groupby(df["DateTime"].dt.date)["LogReturn"].sum()


def create_conditional_returns_heatmap(
    before_market_returns, after_market_returns, threshold, title
):
    before_intervals = ["1m", "5m", "15m", "30m", "1h"]
    after_intervals = ["1m", "5m", "15m", "30m", "1h"]
    data = []

    for before_interval in before_intervals:
        row = []
        for after_interval in after_intervals:
            mask = (before_market_returns[before_interval] >= threshold[0]) & (
                before_market_returns[before_interval] < threshold[1]
            )
            returns = after_market_returns[after_interval][mask].mean()
            row.append(returns)
        data.append(row)

    colors = [
        (0, "rgb(165,0,38)"),  # Dark red for most negative
        (0.25, "rgb(215,48,39)"),  # Red
        (0.45, "rgb(244,109,67)"),  # Light red
        (0.5, "rgb(255,255,255)"),  # White for zero
        (0.55, "rgb(166,217,106)"),  # Light green
        (0.75, "rgb(26,152,80)"),  # Green
        (1, "rgb(0,104,55)"),  # Dark green for most positive
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            x=after_intervals,
            y=before_intervals,
            colorscale=colors,
            zmid=0,
            text=[[f"{val:.2%}" for val in row] for row in data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title, xaxis_title="After Market Open", yaxis_title="Before Market Open"
    )

    return fig


# Calculate returns for different intervals
intervals = {
    "1m": (pd.Timestamp("09:29").time(), pd.Timestamp("09:30").time()),
    "5m": (pd.Timestamp("09:25").time(), pd.Timestamp("09:30").time()),
    "15m": (pd.Timestamp("09:15").time(), pd.Timestamp("09:30").time()),
    "30m": (pd.Timestamp("09:00").time(), pd.Timestamp("09:30").time()),
    "1h": (pd.Timestamp("08:30").time(), pd.Timestamp("09:30").time()),
}

after_intervals = {
    "1m": (pd.Timestamp("09:30").time(), pd.Timestamp("09:31").time()),
    "5m": (pd.Timestamp("09:30").time(), pd.Timestamp("09:35").time()),
    "15m": (pd.Timestamp("09:30").time(), pd.Timestamp("09:45").time()),
    "30m": (pd.Timestamp("09:30").time(), pd.Timestamp("10:00").time()),
    "1h": (pd.Timestamp("09:30").time(), pd.Timestamp("10:30").time()),
}

before_market_returns = {}
after_market_returns = {}

for interval, (start, end) in intervals.items():
    before_market_returns[interval] = calculate_returns(rty_df_filtered, start, end)

for interval, (start, end) in after_intervals.items():
    after_market_returns[interval] = calculate_returns(rty_df_filtered, start, end)

# Create dropdown options
thresholds = [
    (-np.inf, -0.003),
    (-0.003, -0.002),
    (-0.002, -0.001),
    (-0.001, 0),
    (0, 0.001),
    (0.001, 0.002),
    (0.002, 0.003),
    (0.003, np.inf),
]

threshold_names = [
    "Less than -0.3 bps",
    "-0.3 to -0.2 bps",
    "-0.2 to -0.1 bps",
    "-0.1 to 0 bps",
    "0 to +0.1 bps",
    "+0.1 to +0.2 bps",
    "+0.2 to +0.3 bps",
    "More than +0.3 bps",
]

threshold_options = threshold_names + ["Negative", "Positive"]

# Dropdown for selecting before-market return range
selected_threshold = st.selectbox(
    "Select Before-Market Return Range", threshold_options
)

# Create heatmap based on selection
if selected_threshold in threshold_names:
    threshold = thresholds[threshold_names.index(selected_threshold)]
    title = f"Average Returns After Market Open (When Before Market Returns were {selected_threshold})"
elif selected_threshold == "Negative":
    threshold = (-np.inf, 0)
    title = (
        "Average Returns After Market Open (When Before Market Returns were Negative)"
    )
else:  # Positive
    threshold = (0, np.inf)
    title = (
        "Average Returns After Market Open (When Before Market Returns were Positive)"
    )

fig = create_conditional_returns_heatmap(
    before_market_returns, after_market_returns, threshold, title
)
st.plotly_chart(fig)

# Statistical significance
st.subheader("Statistical Significance")
st.write(
    "Performing t-tests to check if the differences in returns are statistically significant."
)

for before_interval in intervals:
    for after_interval in after_intervals:
        if selected_threshold in threshold_names:
            index = threshold_names.index(selected_threshold)
            if index < len(thresholds) - 1:
                returns_1 = after_market_returns[after_interval][
                    (before_market_returns[before_interval] >= thresholds[index][0])
                    & (before_market_returns[before_interval] < thresholds[index][1])
                ]
                returns_2 = after_market_returns[after_interval][
                    (before_market_returns[before_interval] >= thresholds[index + 1][0])
                    & (
                        before_market_returns[before_interval]
                        < thresholds[index + 1][1]
                    )
                ]
                compare_text = (
                    f"Comparing {threshold_names[index]} vs {threshold_names[index+1]}"
                )
            else:
                st.write("No comparison available for the highest threshold.")
                continue
        elif selected_threshold == "Negative":
            returns_1 = after_market_returns[after_interval][
                before_market_returns[before_interval] < 0
            ]
            returns_2 = after_market_returns[after_interval][
                before_market_returns[before_interval] >= 0
            ]
            compare_text = "Comparing Negative vs Non-negative returns"
        else:  # Positive
            returns_1 = after_market_returns[after_interval][
                before_market_returns[before_interval] <= 0
            ]
            returns_2 = after_market_returns[after_interval][
                before_market_returns[before_interval] > 0
            ]
            compare_text = "Comparing Non-positive vs Positive returns"

        if len(returns_1) > 0 and len(returns_2) > 0:
            t_stat, p_value = ttest_ind(returns_1, returns_2)

            st.write(f"Before: {before_interval}, After: {after_interval}")
            st.write(compare_text)
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            st.write(
                "Statistically significant"
                if p_value < 0.05
                else "Not statistically significant"
            )
            st.write("---")

st.write(
    """
This analysis shows how RTY futures perform after market open, conditional on their performance before market open.
The heatmap displays the average returns for different time intervals based on the selected pre-market return range.
The statistical significance tests help us understand if the differences between the selected range and adjacent ranges
are meaningful or potentially due to random chance.
"""
)
