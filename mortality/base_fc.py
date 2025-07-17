import numpy as np
import pandas as pd
import pickle
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from concurrent.futures import ProcessPoolExecutor
import os
import argparse

warnings.filterwarnings("ignore")


def fit_and_forecast(group_df: pd.DataFrame, uid: tuple, target: str, n_samples: int) -> pd.DataFrame:
    group_df = group_df.sort_values("Year").drop_duplicates(subset="Year")
    ts = group_df.set_index("Year")[target].astype(float).sort_index()

    results = []
    max_test_year = ts.index.max()
    max_train_end = max_test_year - 1

    for start_year in range(ts.index.min(), max_train_end - 29 + 1):
        train = ts.loc[start_year: start_year + 29]
        if len(train) < 30 or train.isnull().any():
            continue

        try:
            model = ExponentialSmoothing(train, trend="add", seasonal=None)
            fitted = model.fit()
            fitted_vals = fitted.fittedvalues
            residuals = train.values - fitted_vals

            samples = np.array([
                fitted.forecast(1).values + np.random.normal(0, residuals.std(), size=1)
                for _ in range(n_samples)
            ]).flatten()

            results.append({
                "uid": uid,
                "target": target,
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "test_year": train.index.max() + 1,
                "forecast_samples": samples,
                "residuals": residuals
            })

        except Exception as e:
            print(f"⚠️ Failed for {uid}, {start_year}: {e}")
            results.append({
                "uid": uid,
                "target": target,
                "train_start": train.index.min(),
                "train_end": train.index.max(),
                "test_year": train.index.max() + 1,
                "forecast_samples": np.full(n_samples, np.nan),
                "residuals": np.full(30, np.nan)
            })

    return pd.DataFrame(results)


def force_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, str):
        try:
            return eval(x)
        except Exception:
            return None
    return None


def process_group(args):
    group_df, uid, target, n_samples = args
    group_df = group_df[group_df["gender"] == uid[1]]
    return fit_and_forecast(group_df, uid, target, n_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="../forecasts/forecast_bottom_samples.pkl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv("../data/mortality_complete_data.csv")
    df = df[df["age_group"] == "Total"]
    df = df[df["citizenship"] == "Total"]
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.drop(columns=["age_group", "citizenship"])

    output = {}
    for target in ["deaths", "population"]:
        print(f"🔹 Target: {target}")
        grouped = df.groupby(["Region", "gender"])
        results = []

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_group, (group.copy(), uid, target, args.n_samples))
                for uid, group in grouped
                if group["Year"].nunique() >= 30
            ]
            for future in futures:
                result = future.result()
                if not result.empty:
                    results.append(result)

        combined_df = pd.concat(results, ignore_index=True)
        combined_df["uid"] = combined_df["uid"].apply(force_tuple)
        combined_df = combined_df[combined_df["uid"].notnull()]

        uids = sorted(combined_df["uid"].unique(), key=lambda x: (x[0], x[1]))
        years = sorted(combined_df["test_year"].unique())
        uid_to_idx = {uid: i for i, uid in enumerate(uids)}
        year_to_idx = {year: i for i, year in enumerate(years)}

        samples_arr = np.full((len(uids), len(years), args.n_samples), np.nan)
        residuals_arr = np.full((len(uids), len(years), 30), np.nan)

        for (_, row) in combined_df.iterrows():
            uid, year = row["uid"], row["test_year"]
            if uid in uid_to_idx and year in year_to_idx:
                i, j = uid_to_idx[uid], year_to_idx[year]
                samples_arr[i, j, :] = row["forecast_samples"]
                residuals_arr[i, j, :] = row["residuals"]

        output[target] = {
            "uids": uids,
            "years": years,
            "samples": samples_arr,
            "residuals": residuals_arr,
        }

    with open(args.output, "wb") as f:
        pickle.dump(output, f)

    print(f"✅ Saved forecast samples and residuals to {args.output}")


if __name__ == "__main__":
    main()
