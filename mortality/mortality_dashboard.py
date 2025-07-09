import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Load data
data_path = Path(__file__).resolve().parent.parent / "data" / "mortality_CH.csv"
df = pd.read_csv(data_path)

# Preprocessing
df["Region"] = df["Region"].str.strip().str.capitalize()
df["citizenship"] = df["citizenship"].str.capitalize()
df["Sex"] = df["Sex"].str.capitalize()
df["series"] = df["series"].str.lower()

# Sidebar filters
st.sidebar.title("Filters")
region = st.sidebar.selectbox("Region", sorted(df["Region"].unique()))
sex = st.sidebar.selectbox("Gender", sorted(df["Sex"].unique()))
citizenship = st.sidebar.selectbox("Citizenship", sorted(df["citizenship"].unique()))

# Filter dataset
filtered = df[
    (df["Region"] == region) &
    (df["Sex"] == sex) &
    (df["citizenship"] == citizenship)
]

# Plot all series in 3 vertical subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
series_list = ["death", "population", "mortality_rate"]
titles = ["Deaths", "Population", "Mortality Rate"]

for i, (series_name, title) in enumerate(zip(series_list, titles)):
    subset = filtered[filtered["series"] == series_name]
    sns.lineplot(data=subset, x="Year", y="value", ax=axes[i], marker="o")
    axes[i].set_title(title)
    axes[i].set_ylabel("")
    axes[i].grid(True)

fig.suptitle(f"{region} – Gender: {sex}, Citizenship: {citizenship}", fontsize=14)
st.pyplot(fig)
