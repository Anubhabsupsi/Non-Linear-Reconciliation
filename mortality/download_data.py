import requests
import pandas as pd
import json
from pyjstat import pyjstat
from pathlib import Path

API_URL = "https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0102020000_101/px-x-0102020000_101.px"
QUERY_FILE = "query.json"
OUTPUT_DIR = Path("../data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_and_rename(df, component_name):
    # Standardize canton name
    df["Region"] = df["Canton"].str.split("/").str[0].str.strip()
    df.drop(columns=["Canton"], inplace=True)

    # Rename columns
    df.rename(columns={
        "Citizenship (category)": "citizenship",
        "value": component_name.lower()
    }, inplace=True)

    # Drop 'Demographic component'
    df.drop(columns=["Demographic component"], inplace=True)

    # Safely remove 'No indication' if its value is zero
    if "No indication" in df["Region"].values:
        col = component_name.lower()
        total = df[df["Region"] == "No indication"][col].sum()
        if total == 0:
            print(f"Dropped 'No indication' from {component_name.lower()} (zero total).")
            df = df[df["Region"] != "No indication"]
        else:
            print(f"WARNING: 'No indication' in {component_name.lower()} has non-zero total ({total}) — not dropped.")

    return df


def main():
    # Load query
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        query = json.load(f)

    # Send request
    response = requests.post(API_URL, json=query)
    print("CONGRATULATIONS!🥳 You have successfully downloaded the data. Status:", response.status_code)
    if response.status_code != 200:
        print("OOPS!😔 Error:", response.text)
        return

    df = pyjstat.from_json_stat(response.json())[0]

    # Split by demographic component
    deaths = df[df["Demographic component"] == "Death"].copy()
    pop = df[df["Demographic component"] == "Population on 31 December"].copy()

    # Clean + rename
    deaths = clean_and_rename(deaths, component_name="Death")
    pop = clean_and_rename(pop, component_name="Population")

    # Save deaths and population
    deaths.to_csv(OUTPUT_DIR / "deaths_CH.csv", index=False)
    pop.to_csv(OUTPUT_DIR / "population_CH.csv", index=False)
    print(f"Saved {deaths.shape[0]} death records")
    print(f"Saved {pop.shape[0]} population records")

    # Merge and compute mortality rate
    merged = pd.merge(
        deaths,
        pop,
        on=["Year", "Region", "Sex", "citizenship"],
        how="inner"
    )
    # Merge and compute mortality rate
    merged = pd.merge(
        deaths,
        pop,
        on=["Year", "Region", "Sex", "citizenship"],
        how="inner"
    )
    merged["mortality_rate"] = merged["death"] / merged["population"]

    # Build long format
    long_df = pd.concat([
        deaths.rename(columns={"death": "value"}).assign(series="death"),
        pop.rename(columns={"population": "value"}).assign(series="population"),
        merged[["Year", "Region", "Sex", "citizenship", "mortality_rate"]]
            .rename(columns={"mortality_rate": "value"})
            .assign(series="mortality_rate")
    ], ignore_index=True)

    # Reorder columns
    long_df = long_df[["Year", "Region", "Sex", "citizenship", "series", "value"]]

    # Save to single file
    long_df.to_csv(OUTPUT_DIR / "mortality_CH.csv", index=False)
    print(f"Saved unified long-format dataset with {long_df.shape[0]} rows")

if __name__ == "__main__":
    main()
