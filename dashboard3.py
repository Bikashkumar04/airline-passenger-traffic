# dashboard3.py
# ----------------------------------------------------------
# Airline Passenger Time-Series Dashboard (Streamlit)
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Airline Passenger Time-Series Dashboard", layout="wide")
st.title("✈️ Airline Passenger Time-Series Dashboard")
st.caption("Forecast monthly passenger traffic, visualize trends/seasonality, and monitor KPIs.")

# -------------------------
# Sidebar: data input
# -------------------------
st.sidebar.header("Data & Settings")

uploaded = st.sidebar.file_uploader("Upload CSV (columns: Month, Passengers)", type=["csv"])
date_col = st.sidebar.text_input("Date column name", value="Month")
value_col = st.sidebar.text_input("Value column name", value="Passengers")
forecast_horizon = st.sidebar.number_input("Forecast horizon (months)", 3, 36, 12)
train_ratio = st.sidebar.slider("Train size (%)", 50, 95, 80, step=1)

# Simplified ARIMA configuration
st.sidebar.subheader("ARIMA Configuration")
arima_p = st.sidebar.number_input("p (AR order)", 0, 5, 1)
arima_d = st.sidebar.number_input("d (Differencing)", 0, 2, 1)
arima_q = st.sidebar.number_input("q (MA order)", 0, 5, 1)
use_seasonal = st.sidebar.checkbox("Use Seasonal ARIMA", value=True)
if use_seasonal:
    seasonal_p = st.sidebar.number_input("Seasonal P", 0, 3, 1)
    seasonal_d = st.sidebar.number_input("Seasonal D", 0, 2, 1)
    seasonal_q = st.sidebar.number_input("Seasonal Q", 0, 3, 1)
    seasonal_m = st.sidebar.number_input("Seasonal Period (m)", 1, 24, 12)

# -------------------------
# Load data
# -------------------------
def load_demo():
    return pd.read_csv("airline-passenger-traffic1.csv")

if uploaded:
    df = pd.read_csv(uploaded)
else:
    try:
        df = load_demo()
    except Exception:
        st.warning("No file uploaded and demo file `airline-passenger-traffic1.csv` not found. Please upload a CSV.")
        st.stop()

# Basic validation
if date_col not in df.columns or value_col not in df.columns:
    st.error(f"CSV must contain '{date_col}' and '{value_col}' columns.")
    st.stop()

# Parse dates & sort
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)
df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
df = df.dropna(subset=[value_col])

# -------------------------
# KPIs
# -------------------------
total_obs = len(df)
min_date, max_date = df[date_col].min(), df[date_col].max()
last_val = df[value_col].iloc[-1]

yoy = None
try:
    idx = df.index[-1]
    if idx - 12 >= 0:
        yoy = (df[value_col].iloc[-1] - df[value_col].iloc[idx - 12]) / df[value_col].iloc[idx - 12] * 100
except Exception:
    pass

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Observations", f"{total_obs}", f"{min_date.date()} → {max_date.date()}")
kpi2.metric("Last Month Passengers", f"{int(last_val):,}")
kpi3.metric("YoY Change", f"{yoy:.1f} %" if yoy is not None else "N/A")

# -------------------------
# Plot: Raw series
# -------------------------
st.subheader("📈 Time Series")
fig_ts = px.line(df, x=date_col, y=value_col, title="Monthly Passenger Traffic")
fig_ts.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig_ts, use_container_width=True)

# -------------------------
# Seasonal decomposition
# -------------------------
st.subheader("🧩 Decomposition (Trend / Seasonality / Residual)")
try:
    dec = seasonal_decompose(df[value_col], model="additive", period=12, extrapolate_trend="freq")
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Scatter(x=df[date_col], y=dec.observed, name="Observed"))
    fig_dec.add_trace(go.Scatter(x=df[date_col], y=dec.trend, name="Trend"))
    fig_dec.add_trace(go.Scatter(x=df[date_col], y=dec.seasonal, name="Seasonal"))
    fig_dec.add_trace(go.Scatter(x=df[date_col], y=dec.resid, name="Residual"))
    fig_dec.update_layout(title="Additive Decomposition", height=400, legend_orientation="h")
    st.plotly_chart(fig_dec, use_container_width=True)
except Exception as e:
    st.info(f"Decomposition skipped: {e}")

# -------------------------
# Train/test split
# -------------------------
train_size = int(len(df) * (train_ratio / 100))
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

st.subheader("🧪 Train/Test Split")
st.write(f"Train: {train[date_col].min().date()} → {train[date_col].max().date()}  |  "
         f"Test: {test[date_col].min().date() if len(test) else '—'} → {test[date_col].max().date() if len(test) else '—'}")

# -------------------------
# Fit ARIMA model
# -------------------------
y_train = train.set_index(date_col)[value_col].asfreq("MS").interpolate()

try:
    with st.spinner("Fitting ARIMA model..."):
        if use_seasonal:
            seasonal_order = (seasonal_p, seasonal_d, seasonal_q, seasonal_m)
            sm_model = ARIMA(y_train, order=(arima_p, arima_d, arima_q), 
                            seasonal_order=seasonal_order, freq='MS')
            model_summary_txt = f"SARIMA order=({arima_p},{arima_d},{arima_q}), seasonal_order={seasonal_order}"
        else:
            sm_model = ARIMA(y_train, order=(arima_p, arima_d, arima_q), freq='MS')
            model_summary_txt = f"ARIMA order=({arima_p},{arima_d},{arima_q})"
        
        result = sm_model.fit()
        st.success(f"✅ {model_summary_txt}")
except Exception as e:
    st.error(f"Model fitting failed: {e}")
    st.stop()

# -------------------------
# Forecast
# -------------------------
steps_test = len(test)
steps_future = forecast_horizon

try:
    forecast = result.get_forecast(steps=steps_test + steps_future)
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int(alpha=0.2)

    forecast_index = pd.date_range(y_train.index[-1] + pd.offsets.MonthBegin(), 
                                   periods=steps_test + steps_future, freq="MS")
    pred_df = pd.DataFrame({"Date": forecast_index, "Forecast": pred_mean.values})
    if pred_ci is not None and not pred_ci.empty:
        pred_df["Lower"] = pred_ci.iloc[:, 0].values
        pred_df["Upper"] = pred_ci.iloc[:, 1].values
except Exception as e:
    st.error(f"Forecast generation failed: {e}")
    st.stop()

# -------------------------
# Plot: Actual vs Forecast
# -------------------------
st.subheader("🔮 Actual vs. Forecast")
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode="lines+markers", 
                           name="Actual", line=dict(color="blue")))
fig_fc.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Forecast"], mode="lines+markers",
                            name="Forecast", line=dict(dash="dash", color="red")))
if "Lower" in pred_df:
    fig_fc.add_traces([
        go.Scatter(x=pred_df["Date"], y=pred_df["Upper"], line=dict(width=0), 
                  showlegend=False, hoverinfo='skip'),
        go.Scatter(x=pred_df["Date"], y=pred_df["Lower"], fill="tonexty", line=dict(width=0),
                   name="80% CI", opacity=0.2, fillcolor="rgba(255,0,0,0.2)")
    ])
fig_fc.update_layout(height=420, legend_orientation="h", hovermode='x unified')
st.plotly_chart(fig_fc, use_container_width=True)

# -------------------------
# Error metrics
# -------------------------
if steps_test > 0:
    try:
        test_series = test.set_index(date_col)[value_col].asfreq("MS")
        test_forecast = pred_df.set_index("Date")["Forecast"].iloc[:steps_test]
        test_forecast = test_forecast.reindex(test_series.index)
        
        mae = np.mean(np.abs(test_series - test_forecast))
        rmse = np.sqrt(np.mean((test_series - test_forecast) ** 2))
        mape = np.mean(np.abs((test_series - test_forecast) / test_series)) * 100

        st.subheader("📊 Model Performance Metrics")
        k1, k2, k3 = st.columns(3)
        k1.metric("MAE (Mean Absolute Error)", f"{mae:,.2f}")
        k2.metric("RMSE (Root Mean Squared Error)", f"{rmse:,.2f}")
        k3.metric("MAPE (Mean Absolute %)", f"{mape:.2f}%")
    except Exception as e:
        st.warning(f"Could not calculate error metrics: {e}")

# -------------------------
# Download forecast table
# -------------------------
st.subheader("📥 Download Forecast Table")
out_df = pred_df.rename(columns={"Date": "Month"})
st.dataframe(out_df.tail(forecast_horizon), use_container_width=True)
st.download_button(
    "📥 Download full forecast as CSV",
    data=out_df.to_csv(index=False),
    file_name="airline_passenger_forecast.csv",
    mime="text/csv",
)

# -------------------------
# Model Summary (Expandable)
# -------------------------
with st.expander("📋 View Model Summary"):
    st.text(str(result.summary()))