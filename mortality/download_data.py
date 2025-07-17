import requests
import json
import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
from pyjstat import pyjstat

API_URL = "https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0102020000_104/px-x-0102020000_104.px"
AGE_CODES = [str(i) for i in range(0, 101)]
BATCH_SIZE = 2
QUERY_PATH = "query.json"
SLEEP = 1  # seconds
MAX_WORKERS =  max(1, os.cpu_count() - 1) # Adjust based on your system and BFS limits

def load_query():
    with open(QUERY_PATH, "r") as f:
        return json.load(f)

def fetch(component_code, ages, query_base):
    q = json.loads(json.dumps(query_base))  # deep copy
    for dim in q["query"]:
        if dim["code"] == "Alter":
            dim["selection"]["values"] = ages
        elif dim["code"] == "Demografische Komponente":
            dim["selection"]["values"] = [component_code]

    try:
        r = requests.post(API_URL, json=q, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            tqdm.write(f"❌ HTTP {r.status_code} for ages {ages} and component {component_code}")
            return None
        df = pyjstat.from_json_stat(r.json())[0]
        df["component"] = "deaths" if component_code == "2" else "population"
        return df
    except Exception as e:
        tqdm.write(f"❌ JSON parsing error for ages {ages} and component {component_code}: {e}")
        return None

def clean_and_rename(df):
    print("🧼 Cleaning and standardizing column names...")

    # Clean canton name
    df["Region"] = df["Canton"].str.split("/").str[0].str.strip()
    df.drop(columns=["Canton"], inplace=True)

    # Rename columns
    df.rename(columns={
        "Sex": "gender",
        "Citizenship (category)": "citizenship"
    }, inplace=True)

    # Drop unused column
    df = df.drop(columns=["Demographic component"], errors="ignore")

    # Remove 'No indication' if total is zero
    if "No indication" in df["Region"].values:
        total = df.loc[df["Region"] == "No indication", "value"].sum()
        if total == 0:
            df = df[df["Region"] != "No indication"]

    # Filter out existing totals from BFS
    df = df[df["gender"] != "Sex - total"]
    df = df[df["citizenship"] != "Citizenship (category) - total"]
    df = df[df["Age"] != "Age - total"]

    return df



def group_ages(df):
    print("------------------------------------------------------------------")
    print("📊 Grouping ages into 5-year bins, with '75+' as the top group...")
    print("------------------------------------------------------------------")

    def map_age(age_str):
        if age_str == "Total":
            return "Total"
        if age_str == "99 years or older":
            return "75+"
        try:
            age = int(age_str.split()[0])
            if age >= 75:
                return "75+"
            lower = (age // 5) * 5
            upper = lower + 4
            return f"{lower}–{upper}"
        except Exception:
            return "unknown"

    df["age_group"] = df["Age"].apply(map_age)
    return df

def add_gender_totals(df):
    print("➕ Adding 'Total' gender rows...")
    base_df = df[df["gender"] != "Total"]
    grouped = base_df.groupby(
        ["Year", "Region", "citizenship", "age_group", "component"], as_index=False
    )["value"].sum()
    grouped["gender"] = "Total"
    return pd.concat([df, grouped], ignore_index=True)


def add_national_total(df):
    print("🇨🇭 Adding Switzerland-wide totals...")
    base_df = df[df["Region"] != "Switzerland"]
    national = base_df.groupby(
        ["Year", "citizenship", "gender", "age_group", "component"], as_index=False
    )["value"].sum()
    national["Region"] = "Switzerland"
    return pd.concat([df, national], ignore_index=True)


def add_age_totals(df):
    print("➕ Adding 'Total' age group rows...")
    base_df = df[df["age_group"] != "Total"]
    grouped = base_df.groupby(
        ["Year", "Region", "citizenship", "gender", "component"], as_index=False
    )["value"].sum()
    grouped["age_group"] = "Total"
    return pd.concat([df, grouped], ignore_index=True)

def add_citizenship_totals(df):
    print("➕ Adding 'Total' citizenship rows...")
    base_df = df[df["citizenship"] != "Total"]
    grouped = base_df.groupby(
        ["Year", "Region", "gender", "age_group", "component"], as_index=False
    )["value"].sum()
    grouped["citizenship"] = "Total"
    return pd.concat([df, grouped], ignore_index=True)


def compute_mortality_rate(df):
    print("------------------------------------------------------------------")
    print("📈 Computing mortality rates by age group...")
    print("------------------------------------------------------------------")

    df_wide = df.pivot_table(
        index=["Year", "Region", "citizenship", "gender", "age_group"],
        columns="component",
        values="value",
        aggfunc="sum"
    ).reset_index()

    if "deaths" not in df_wide.columns or "population" not in df_wide.columns:
        raise ValueError("❌ Missing required columns to compute mortality rate.")

    df_wide["mortality_rate"] = df_wide["deaths"] / df_wide["population"]
    return df_wide

def main():
    raw_path = "../data/mortality_raw_data.csv"
    output_path = "../data/mortality_complete_data.csv"

    # Skip download if raw file exists
    if os.path.exists(raw_path):
        print(f"📂 Raw file already exists at {raw_path}, skipping download...")
        final = pd.read_csv(raw_path)
    else:
        print("📥 Starting download from BFS API...")
        print("------------------------------------------------------------")
        print("Huge dataset, might take a while...")
        print("------------------------------------------------------------")

        query_base = load_query()
        fetch_partial = partial(fetch, query_base=query_base)
        age_batches = [AGE_CODES[i:i + BATCH_SIZE] for i in range(0, len(AGE_CODES), BATCH_SIZE)]
        jobs = [(comp, ages) for comp in ["2", "14"] for ages in age_batches]

        all_dfs = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(fetch_partial, comp, ages): (comp, ages)
                       for comp, ages in jobs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading data"):
                try:
                    df = future.result()
                    if df is not None:
                        all_dfs.append(df)
                except Exception as e:
                    tqdm.write(f"❌ Future failed with exception: {e}")

        print("✅ Success 🥳: Downloaded data")

        print("📚 Combining all data chunks...")
        final = pd.concat(all_dfs, ignore_index=True)

        print(f"💾 Saving raw data to {raw_path}")
        final.to_csv(raw_path, index=False)

    print("🧹 Cleaning data...")
    cleaned = clean_and_rename(final)

    print("📦 Grouping into age bins...")
    grouped = group_ages(cleaned)
    grouped = grouped.drop(columns=["Age"])

    # 🧹 Remove all pre-aggregated totals
    grouped = grouped[
        (grouped["age_group"] != "Total") &
        (grouped["gender"] != "Total") &
        (grouped["Region"] != "Switzerland")
        ]

    # ✅ Add custom totals in correct order
    grouped = add_gender_totals(grouped)
    grouped = add_national_total(grouped)
    grouped = add_age_totals(grouped)
    grouped = add_citizenship_totals(grouped)

    print("⚙️ Calculating mortality rate...")
    mortality_data = compute_mortality_rate(grouped)
    print(mortality_data[(mortality_data['Region'] == 'Switzerland') &
                         (mortality_data['gender'] == 'Total') &
                         (mortality_data['age_group'] == 'Total') &
                         (mortality_data['citizenship'] == 'Total')].iloc[-1])

    print(f"💾 Saving processed data to {output_path}")
    mortality_data.to_csv(output_path, index=False)
    print("✅ Done! Data saved.")

if __name__ == "__main__":
    main()
