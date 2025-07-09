import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
DATA_PATH = Path("../data/mortality_CH.csv")
df = pd.read_csv(DATA_PATH)

# Normalize columns
df["Region"] = df["Region"].str.strip().str.capitalize()
df["citizenship"] = df["citizenship"].str.capitalize()
df["Sex"] = df["Sex"].str.capitalize()
df["series"] = df["series"].str.lower()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Targets
target_regions = ["Switzerland", "Zürich", "Genève", "Ticino"]
target_series = ["death", "population", "mortality_rate"]

# Output dir
output_dir = Path("../figures")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

### 1. By Region only (all regions)
for series in target_series:
    target_regions = ["Switzerland", "Zürich", "Genève", "Ticino"]
    plt.figure(figsize=(12, 8))
    subset = df[(df["series"] == series) & (df["Region"].isin(target_regions))]
    sns.lineplot(data=subset, x="Year", y="value", hue="Region", errorbar=None, marker="o")
    plt.title(f"{series.capitalize()} by Region (All)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{series}_all_regions.png")
    plt.close()

### 2. By Gender only (aggregated over all regions)
for series in target_series:
    subset = df[df["series"] == series].groupby(["Year", "Sex", "series"], as_index=False)["value"].sum()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=subset, x="Year", y="value", hue="Sex", marker="o",errorbar=None)
    plt.title(f"{series.capitalize()} by Gender (All Regions)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{series}_all_by_gender.png")
    plt.close()

### 3. By Citizenship only
for series in target_series:
    subset = df[df["series"] == series].groupby(["Year", "citizenship", "series"], as_index=False)["value"].sum()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=subset, x="Year", y="value", hue="citizenship", marker="o",errorbar=None)
    plt.title(f"{series.capitalize()} by Citizenship (All Regions)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{series}_all_by_citizenship.png")
    plt.close()

### 4. Region + Gender
for region in target_regions:
    for series in target_series:
        subset = df[(df["Region"] == region) & (df["series"] == series)]
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=subset, x="Year", y="value", hue="Sex", marker="o",errorbar=None)
        plt.title(f"{series.capitalize()} in {region} by Gender")
        plt.tight_layout()
        plt.savefig(output_dir / f"{series}_{region}_by_gender.png")
        plt.close()

### 5. Region + Citizenship
for region in target_regions:
    for series in target_series:
        subset = df[(df["Region"] == region) & (df["series"] == series)]
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=subset, x="Year", y="value", hue="citizenship", marker="o",errorbar=None)
        plt.title(f"{series.capitalize()} in {region} by Citizenship")
        plt.tight_layout()
        plt.savefig(output_dir / f"{series}_{region}_by_citizenship.png")
        plt.close()

### 6. Region + Gender + Citizenship
for region in target_regions:
    for series in target_series:
        subset = df[(df["Region"] == region) & (df["series"] == series)]
        g = sns.FacetGrid(
            subset,
            col="Sex",
            row="citizenship",
            margin_titles=True,
            height=3.5,
            aspect=1.6,
        )
        g.map(sns.lineplot, "Year", "value", marker="o", color="steelblue",errorbar=None)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f"{series.capitalize()} in {region} by Gender & Citizenship")
        g.savefig(output_dir / f"{series}_{region}_by_gender_citizenship.png")
        plt.close()
