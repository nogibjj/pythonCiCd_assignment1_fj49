import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
@st.cache_data
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

# Merge datasets
merged_df = pd.merge(rty_df, spx_df, on="DateTime", suffixes=("_rty", "_spx"))
merged_df = pd.merge(merged_df, volume_df, on="DateTime", how="left")


# Feature engineering
def create_features(df):
    df["prev_minute_return_rty"] = df.groupby("Date")["LogReturn_rty"].shift(1)
    df["prev_minute_return_spx"] = df.groupby("Date")["LogReturn_spx"].shift(1)
    df["prev_20min_volatility"] = (
        df.groupby("Date")["LogReturn_rty"].rolling(20).std().reset_index(0, drop=True)
    )
    df["hour"] = df["DateTime"].dt.hour
    df["day_of_week"] = df["DateTime"].dt.dayofweek
    df["day_of_month"] = df["DateTime"].dt.day
    df["month"] = df["DateTime"].dt.month
    df["year"] = df["DateTime"].dt.year
    df["ewma"] = df.groupby("Date")["LogReturn_rty"].transform(
        lambda x: x.ewm(span=20).mean()
    )
    df["volume_change"] = df.groupby("Date")["volume"].pct_change()
    df["rty_spx_spread"] = df["LogReturn_rty"] - df["LogReturn_spx"]
    return df


merged_df = create_features(merged_df)

# Prepare data for modeling
features = [
    "prev_minute_return_rty",
    "prev_minute_return_spx",
    "prev_20min_volatility",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "ewma",
    "volume_change",
    "rty_spx_spread",
]

X = merged_df[features].dropna()
y = merged_df["LogReturn_rty"].iloc[
    1:
]  # Shift target by 1 to predict next minute return

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("RTY Next Minute Return Prediction")

st.header("Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.6f}")
col2.metric("R-squared", f"{r2:.6f}")

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": features, "importance": model.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)

st.header("Feature Importance")
fig_importance = px.bar(
    feature_importance, x="importance", y="feature", orientation="h"
)
st.plotly_chart(fig_importance)

# Actual vs Predicted plot
st.header("Actual vs Predicted Returns")
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Actual"))
fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_pred, mode="lines", name="Predicted"))
fig_pred.update_layout(
    title="Actual vs Predicted Returns", xaxis_title="Date", yaxis_title="Return"
)
st.plotly_chart(fig_pred)

# Backtesting
st.header("Backtesting")

# Simple trading strategy: Buy if predicted return is positive, sell if negative
y_pred_all = model.predict(X)
merged_df.loc[X.index, "predicted_return"] = y_pred_all
merged_df["position"] = np.where(merged_df["predicted_return"] > 0, 1, -1)
merged_df["strategy_return"] = (
    merged_df["position"].shift(1) * merged_df["LogReturn_rty"]
)

# Calculate cumulative returns
merged_df["cumulative_return_rty"] = (1 + merged_df["LogReturn_rty"]).cumprod()
merged_df["cumulative_return_strategy"] = (1 + merged_df["strategy_return"]).cumprod()

# Plot cumulative returns
fig_backtest = go.Figure()
fig_backtest.add_trace(
    go.Scatter(
        x=merged_df.index,
        y=merged_df["cumulative_return_rty"],
        mode="lines",
        name="Buy and Hold",
    )
)
fig_backtest.add_trace(
    go.Scatter(
        x=merged_df.index,
        y=merged_df["cumulative_return_strategy"],
        mode="lines",
        name="Strategy",
    )
)
fig_backtest.update_layout(
    title="Backtesting Results", xaxis_title="Date", yaxis_title="Cumulative Return"
)
st.plotly_chart(fig_backtest)

# Performance metrics
total_return_bh = merged_df["cumulative_return_rty"].iloc[-1] - 1
total_return_strategy = merged_df["cumulative_return_strategy"].iloc[-1] - 1
sharpe_ratio_bh = (
    np.sqrt(252) * merged_df["LogReturn_rty"].mean() / merged_df["LogReturn_rty"].std()
)
sharpe_ratio_strategy = (
    np.sqrt(252)
    * merged_df["strategy_return"].mean()
    / merged_df["strategy_return"].std()
)

st.subheader("Performance Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Return (Buy and Hold)", f"{total_return_bh:.2%}")
col2.metric("Total Return (Strategy)", f"{total_return_strategy:.2%}")
col1.metric("Sharpe Ratio (Buy and Hold)", f"{sharpe_ratio_bh:.2f}")
col2.metric("Sharpe Ratio (Strategy)", f"{sharpe_ratio_strategy:.2f}")

# Prediction for next minute
st.header("Prediction for Next Minute")
last_data = merged_df.iloc[-1][features]
next_minute_pred = model.predict([last_data])[0]
st.metric("Predicted Return for Next Minute", f"{next_minute_pred:.6f}")

# Allow user to input custom values for prediction
st.header("Custom Prediction")
custom_input = {}
for feature in features:
    custom_input[feature] = st.number_input(
        f"Enter value for {feature}", value=float(last_data[feature])
    )

custom_pred = model.predict([list(custom_input.values())])[0]
st.metric("Custom Prediction", f"{custom_pred:.6f}")
