import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="RTY Futures ML Model")


# Load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, nrows=100000)

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

st.title("RTY Futures Returns Prediction Model")

st.write(
    """
This application presents a machine learning model for predicting RTY futures returns.
We'll walk through the process of feature engineering, model training, and evaluation.
"""
)

# Feature Engineering
st.header("1. Feature Engineering")

st.write(
    """
We start by creating features that might be predictive of future returns.
These include technical indicators, lagged returns, and rolling statistics.
"""
)

# Merge datasets
volume_df["NormalizedVolume"] = volume_df["volume"] / volume_df["volume"].sum()
merged_df = pd.merge(rty_df, volume_df, on="DateTime", how="left")
merged_df["NormalizedVolume"].fillna(0, inplace=True)
merged_df["LogReturnRTY"] = merged_df["LogReturn"]
merged_df_final = pd.merge(merged_df, spx_df, on="DateTime", how="left")

# Feature engineering
merged_df_final["Hour"] = merged_df_final["DateTime"].dt.hour
merged_df_final["Minute"] = merged_df_final["DateTime"].dt.minute
merged_df_final["prev_20_min_vol"] = merged_df_final["LogReturnRTY"].rolling(
    window=20
).std() * np.sqrt(252)

lamb = 0.94  # smoothing parameter
merged_df_final["ewma"] = merged_df_final["prev_20_min_vol"] * lamb + (
    1 - lamb
) * merged_df_final.prev_20_min_vol.shift(1)

final_df = merged_df_final[
    [
        "DateTime",
        "LogReturnRTY",
        "LogReturn_y",
        "Hour",
        "Minute",
        "prev_20_min_vol",
        "tradeCount",
        "volume",
        "volume_vwap",
        "ewma",
    ]
]
final_df["prev_return_rty"] = final_df.LogReturnRTY.shift(1)
final_df["prev_return_spx"] = final_df.LogReturn_y.shift(1)
final_df["prev_volume"] = final_df.volume.shift(1)
final_df["tradeCount"] = final_df.tradeCount.shift(1)
final_df["Day"] = final_df["DateTime"].dt.day
final_df["Month"] = final_df["DateTime"].dt.month
final_df["Year"] = final_df["DateTime"].dt.year

final_df = final_df[
    [
        "LogReturnRTY",
        "Hour",
        "Minute",
        "prev_20_min_vol",
        "prev_return_rty",
        "prev_return_spx",
        "prev_volume",
        "Day",
        "Month",
        "Year",
        "ewma",
    ]
]

st.write("Here's a sample of our engineered features:")
st.dataframe(final_df.head())

# Model Training
st.header("2. Model Training")

st.write(
    """
We use a Random Forest Regressor to predict future returns.
This model can capture non-linear relationships and is less prone to overfitting.
"""
)

# Prepare data for modeling
features = [
    "Hour",
    "Minute",
    "prev_20_min_vol",
    "prev_return_rty",
    "prev_return_spx",
    "prev_volume",
    "Day",
    "Month",
    "Year",
    "ewma",
]
target = "LogReturnRTY"

model_data = final_df[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

st.dataframe(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Model Evaluation
st.header("3. Model Evaluation")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error: {mse:.6f}")
st.write(f"R-squared Score: {r2:.6f}")

# Plotting actual vs predicted returns
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Actual Returns"))
fig.add_trace(
    go.Scatter(x=y_test.index, y=y_pred, mode="lines", name="Predicted Returns")
)
fig.update_layout(
    title="Actual vs Predicted Returns",
    xaxis_title="Date",
    yaxis_title="Returns",
    template="plotly_white",
)
st.plotly_chart(fig)

# Feature Importance
st.header("4. Feature Importance")

st.write(
    """
Let's examine which features are most important in making predictions.
This can give us insights into what drives RTY futures returns.
"""
)

feature_importance = pd.DataFrame(
    {"feature": features, "importance": rf.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)

fig = px.bar(
    feature_importance, x="feature", y="importance", title="Feature Importance"
)
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)

# Strategy Comparison
st.header("5. Strategy Comparison")

st.write(
    """
Finally, let's compare our model's performance to a simple buy-and-hold strategy.
"""
)

# Calculate cumulative returns
ans_df = pd.DataFrame({"Actual": y_test, "Predicted_rf": y_pred})
X_test_with_pred = X_test.join(ans_df)

conditions = [
    X_test_with_pred["Predicted_rf"] >= 0.00005,
    X_test_with_pred["Predicted_rf"] <= -0.00005,
    (X_test_with_pred["Predicted_rf"] > -0.00005)
    & (X_test_with_pred["Predicted_rf"] < 0.00005),
]
values = ["Buy", "Sell", "Hold"]
X_test_with_pred["Decision"] = np.select(conditions, values)

# Calculate strategy returns
X_test_with_pred["StrategyReturn"] = np.where(
    X_test_with_pred["Decision"] == "Buy",
    X_test_with_pred["Actual"],
    np.where(X_test_with_pred["Decision"] == "Sell", -X_test_with_pred["Actual"], 0),
)

cumulative_actual = (1 + X_test_with_pred["Actual"]).cumprod()
cumulative_strategy = (1 + X_test_with_pred["StrategyReturn"]).cumprod()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=cumulative_actual.index,
        y=cumulative_actual,
        mode="lines",
        name="Buy and Hold",
    )
)
fig.add_trace(
    go.Scatter(
        x=cumulative_strategy.index,
        y=cumulative_strategy,
        mode="lines",
        name="Model Strategy",
    )
)
fig.update_layout(
    title="Cumulative Returns: Model vs Buy-and-Hold",
    xaxis_title="Date",
    yaxis_title="Cumulative Returns",
    template="plotly_white",
)
st.plotly_chart(fig)

# Conclusion
st.header("6. Conclusion")

st.write(
    f"""
Our Random Forest model shows promise in predicting RTY futures returns.
Key findings:
1. The model achieves an R-squared score of {r2:.4f}, indicating it explains {r2*100:.2f}% of the variance in returns.
2. The most important features for prediction are {', '.join(feature_importance['feature'].head(3).tolist())}.
3. The model strategy {"outperforms" if cumulative_strategy.iloc[-1] > cumulative_actual.iloc[-1] else "underperforms"} a simple buy-and-hold strategy over the test period.

Areas for further investigation:
1. Feature engineering: Can we create more predictive features?
2. Model tuning: Can we improve performance with hyperparameter optimization?
3. Alternative models: How do other algorithms perform on this task?
"""
)

st.write("Thank you for exploring this RTY Futures Returns Prediction Model!")
