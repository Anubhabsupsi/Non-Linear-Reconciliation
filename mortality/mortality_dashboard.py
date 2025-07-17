import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Load data
data_path = Path(__file__).resolve().parent.parent / "data" / "mortality_complete_data.csv"
df = pd.read_csv(data_path)

# Preprocessing
df["Region"] = df["Region"].str.strip()
df["citizenship"] = df["citizenship"].str.capitalize()
df["gender"] = df["gender"].str.capitalize()
df["age_group"] = df["age_group"].replace("unknown", "Unknown").str.strip()

# Sidebar filters
st.sidebar.title("🔍 Filters")

region = st.sidebar.selectbox("📍 Region", sorted(df["Region"].unique()))
sex = st.sidebar.selectbox("🚻 Gender", sorted(df["gender"].unique()))
citizenship = st.sidebar.selectbox("🛂 Citizenship", sorted(df["citizenship"].unique()))

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("📅 Year range", year_min, year_max, (year_min, year_max))

st.sidebar.markdown("👶 **Select Age Group(s)**")

age_group_options = sorted(df["age_group"].dropna().unique())

# Define number of columns for layout
cols = st.sidebar.columns(3)  # 3 columns

selected_ages = []
for i, age_group in enumerate(age_group_options):
    if cols[i % 3].checkbox(age_group, value=True):
        selected_ages.append(age_group)

# Filter dataset
filtered = df[
    (df["Region"] == region) &
    (df["gender"] == sex) &
    (df["citizenship"] == citizenship) &
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1]) &
    (df["age_group"].isin(selected_ages))
]

if filtered.empty:
    st.warning("No data found for this combination.")
    st.stop()

# Melt to long format for plotting
long_df = filtered.melt(
    id_vars=["Year", "Region", "citizenship", "gender", "age_group"],
    value_vars=["deaths", "population", "mortality_rate"],
    var_name="series",
    value_name="value"
)

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
series_list = ["deaths", "population", "mortality_rate"]
titles = ["Deaths", "Population", "Mortality Rate"]

for i, (series_name, title) in enumerate(zip(series_list, titles)):
    subset = long_df[long_df["series"] == series_name]
    sns.lineplot(
        data=subset,
        x="Year",
        y="value",
        hue="age_group",
        ax=axes[i],
        marker="o"
    )
    axes[i].set_title(title)
    axes[i].set_ylabel("")
    axes[i].legend(title="Age group", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[i].grid(True)

fig.suptitle(f"{region} – Gender: {sex}, Citizenship: {citizenship}", fontsize=16)
fig.tight_layout()
st.pyplot(fig)
