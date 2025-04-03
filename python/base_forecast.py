import pandas as pd
import os
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.models import AutoETS
from typing import Dict, Tuple

def fit_model(data: pd.DataFrame, choice: str = 'arima', h: int = 10, window_size: int = 300, step: int = 10) -> Dict[str, pd.DataFrame]:
    dates = pd.date_range(start='2000-01-01', periods=len(data), freq='D')
    df_long = pd.concat([
        pd.DataFrame({
            'unique_id': name,
            'ds': dates,
            'y': data[name].values
        })
        for name in data.columns
    ])

    if choice == 'arima':
        model_cls = AutoARIMA
        model_name = 'AutoARIMA'
    elif choice == 'ets':
        model_cls = lambda: AutoETS(season_length=1)
        model_name = 'AutoETS'
    else:
        raise ValueError(f"Invalid choice: {choice}. Available choices are 'arima' and 'ets'.")

    forecast_dict = {}

    for uid in df_long['unique_id'].unique():
        print(f"Fitting {model_name} for {uid}")
        df_series = df_long[df_long['unique_id'] == uid].reset_index(drop=True)
        rows_forecast = []

        for start in range(0, len(df_series) - window_size - h + 1, step):
            train = df_series.iloc[start:start + window_size]
            test_start = pd.to_datetime(df_series['ds'].iloc[start + window_size])

            forecast = StatsForecast(models=[model_cls()], freq='D').forecast(
                df=train[['unique_id', 'ds', 'y']], h=h
            )

            forecast['ds'] = pd.to_datetime(forecast['ds'])
            forecast['horizon'] = (forecast['ds'] - test_start).dt.days + 1
            forecast = forecast[(forecast['horizon'] >= 1) & (forecast['horizon'] <= h)]

            row = pd.Series(index=[f"h={i}" for i in range(1, h + 1)], dtype='float64')
            row.update(
                forecast.set_index(forecast['horizon'].map(lambda x: f"h={x}"))[model_name]
            )
            row.name = test_start
            rows_forecast.append(row.to_frame().T)

        forecast_dict[uid] = pd.concat(rows_forecast)

    return forecast_dict


def get_test_dict(data: pd.DataFrame, h: int = 10, window_size: int = 300, step: int = 10) -> Dict[str, pd.DataFrame]:
    dates = pd.date_range(start='2000-01-01', periods=len(data), freq='D')
    df_long = pd.concat([
        pd.DataFrame({
            'unique_id': name,
            'ds': dates,
            'y': data[name].values
        })
        for name in data.columns
    ])

    test_dict = {}

    for uid in df_long['unique_id'].unique():
        df_series = df_long[df_long['unique_id'] == uid].reset_index(drop=True)
        rows_truth = []

        for start in range(0, len(df_series) - window_size - h + 1, step):
            test_start = pd.to_datetime(df_series['ds'].iloc[start + window_size])
            actual_values = df_series['y'].iloc[start + window_size : start + window_size + h].values
            actual_row = pd.Series(actual_values, index=[f"h={i}" for i in range(1, h + 1)], dtype='float64')
            actual_row.name = test_start
            rows_truth.append(actual_row.to_frame().T)

        test_dict[uid] = pd.concat(rows_truth)

    return test_dict

def safe_pickle(obj, path: str):
    if not os.path.exists(path):
        pd.to_pickle(obj, path)
        print(f"✅ Saved: {path}")
    else:
        print(f"⚠️ Skipped (already exists): {path}")

def main():
    data_folder = '../data/'
    fc_folder = '../forecasts/'
    os.makedirs(fc_folder, exist_ok=True)

    # Load data
    data_indep = pd.read_pickle(os.path.join(data_folder, 'indep_ar_process.pkl'))
    data_corr = pd.read_pickle(os.path.join(data_folder, 'corr_ar_process.pkl'))

    choice = 'arima'

    # Independent data
    base_fc_indep = fit_model(data_indep, choice=choice)
    test_data_indep = get_test_dict(data_indep)

    safe_pickle(base_fc_indep, os.path.join(fc_folder, f'indep_base_fc_{choice}.pkl'))
    safe_pickle(test_data_indep, os.path.join(fc_folder, 'indep_test_dict.pkl'))

    # Correlated data
    base_fc_corr = fit_model(data_corr, choice=choice)
    test_data_corr = get_test_dict(data_corr)

    safe_pickle(base_fc_corr, os.path.join(fc_folder, f'corr_base_fc_{choice}.pkl'))
    safe_pickle(test_data_corr, os.path.join(fc_folder, 'corr_test_dict.pkl'))


if __name__ in '__main__':
    main()