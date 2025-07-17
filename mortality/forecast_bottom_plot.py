import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import streamlit as st

# -----------------------------
# 📂 Load Forecast File
# -----------------------------
@st.cache_data
def load_forecast_data():
    forecast_path = (Path(__file__).resolve().parent.parent / "forecasts" / "forecast_bottom_samples.pkl").resolve()
    with open(forecast_path, "rb") as f:
        return pickle.load(f)

forecast_data = load_forecast_data()

# -----------------------------
# 📂 Load Ground Truth Data
# -----------------------------
data_path = (Path(__file__).resolve().parent.parent / "data" / "mortality_complete_data.csv").resolve()
df = pd.read_csv(data_path)

# Preprocessing
df["Region"] = df["Region"].str.strip()
df["citizenship"] = df["citizenship"].str.capitalize()
df["gender"] = df["gender"].str.capitalize()
df["age_group"] = df["age_group"].replace("unknown", "Unknown").str.strip()

# -----------------------------
# 🎛 Sidebar Options
# -----------------------------
st.sidebar.title("🔧 Forecast Viewer")
target = st.sidebar.selectbox("📊 Target Variable", list(forecast_data.keys()))

uids = forecast_data[target]["uids"]
regions = sorted(set(uid[0] for uid in uids))
genders = sorted(set(uid[1] for uid in uids))

region = st.sidebar.selectbox("📍 Region", regions)
gender = st.sidebar.selectbox("🚻 Gender", genders)

# -----------------------------
# 📈 Extract Forecast
# -----------------------------
years = forecast_data[target]["years"]
samples = forecast_data[target]["samples"]
uid_index = uids.index((region, gender))
forecast_samples = samples[uid_index]  # shape (n_years, n_samples)

mean_forecast = np.nanmean(forecast_samples, axis=1)
lower = np.nanpercentile(forecast_samples, 2.5, axis=1)
upper = np.nanpercentile(forecast_samples, 97.5, axis=1)

# -----------------------------
# 📉 Ground Truth
# -----------------------------
truth = df[
    (df["Region"] == region) &
    (df["gender"] == gender) &
    (df["citizenship"] == "Total") &
    (df["age_group"] == "Total")
].sort_values("Year")

truth_series = truth.set_index("Year")[target].astype(float)
first_forecast_year = years[0]

truth_before = truth_series[truth_series.index < first_forecast_year]
truth_after = truth_series[truth_series.index >= first_forecast_year]

# -----------------------------
# 📊 Plotting
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(truth_before.index, truth_before.values, label="Ground Truth (Train)", marker="o", color="black")
ax.plot(truth_after.index, truth_after.values, label="Ground Truth (Test)", marker="o", linestyle="--", color="gray")

ax.plot(years, mean_forecast, label="Forecast Mean", marker="o", color="tab:blue")
ax.fill_between(years, lower, upper, alpha=0.3, label="95% CI", color="tab:blue")

ax.set_title(f"{target.capitalize()} Forecast – {region} ({gender})", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel(target.capitalize())
ax.grid(True)
ax.legend()

st.pyplot(fig)
