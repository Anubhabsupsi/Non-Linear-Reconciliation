import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def fit_base_forecast_rolling(data, name='', window_size=50):
    years = data['Year'].values
    y = data['Total'].reset_index(drop=True)  # ensure simple numeric index
    split = int(len(y) * 0.8)
    test = y[split:]
    years_test = years[split:]

    # Store outputs
    point_forecasts = []
    lower_bounds = []
    upper_bounds = []
    all_samples = []

    for t in range(len(test)):
        window_start = split + t - window_size
        window_end = split + t

        if window_start < 0:
            print(f"Skipping step {t} — not enough data for window size {window_size}")
            point_forecasts.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            all_samples.append(np.full(100, np.nan))
            continue

        train_window = y[window_start:window_end]

        try:
            model = ExponentialSmoothing(train_window, trend='add', seasonal=None)
            model_fit = model.fit()

            yhat = model_fit.forecast(1)[0]
            point_forecasts.append(yhat)

            residuals = model_fit.fittedvalues - train_window
            if len(residuals) == 0 or np.all(residuals == 0):
                raise ValueError("Residuals are empty or constant")

            step_samples = yhat + np.random.choice(residuals, size=100, replace=True)
            lower_bounds.append(np.percentile(step_samples, 5))
            upper_bounds.append(np.percentile(step_samples, 95))
            all_samples.append(step_samples)

        except Exception as e:
            print(f"Step {t}: Forecast failed — {e}")
            point_forecasts.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            all_samples.append(np.full(100, np.nan))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(years, y.values, label='True Series', color='black')
    plt.plot(years_test, point_forecasts, '--', label='Point Forecast', color='blue')
    plt.plot(years_test, test.values, ':', label='Actual', color='green')
    plt.fill_between(years_test, lower_bounds, upper_bounds, color='blue', alpha=0.4, label='90% Predictive Interval')
    plt.xlabel('Year')
    plt.ylabel(name)
    plt.title(f'Sliding Forecast for {name} (window={window_size}) with Exponential Smoothing + Residual Bootstrap')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return point_forecasts, np.array(all_samples)

def fit_base_forecast(data,name=''):
    years = data['Year'].values
    y = data['Total']
    split = int(len(y) * 0.8)
    train, test = y.iloc[:split], y.iloc[split:]
    years_test = years[split:]

    # Store outputs
    point_forecasts = []
    lower_bounds = []
    upper_bounds = []
    all_samples = []

    for t in range(len(test)):
        try:
            model = ExponentialSmoothing(train, trend='add', seasonal=None)
            model_fit = model.fit()

            yhat = model_fit.forecast(1)[0]
            point_forecasts.append(yhat)

            residuals = model_fit.fittedvalues - train
            if len(residuals) == 0 or np.all(residuals == 0):
                raise ValueError("Residuals are empty or constant")

            step_samples = yhat + np.random.choice(residuals, size=100, replace=True)

            lower_bounds.append(np.percentile(step_samples, 5))
            upper_bounds.append(np.percentile(step_samples, 95))
            all_samples.append(step_samples)

        except Exception as e:
            print(f"Step {t}: Forecast failed — {e}")
            point_forecasts.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            all_samples.append(np.full(100, np.nan))

        # Use iloc[t] instead of test[t]
        train = pd.concat([train, pd.Series([test.iloc[t]], index=[test.index[t]])])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(years, y.values, label='True Series', color='black')
    plt.plot(years_test, point_forecasts, '--', label='Point Forecast', color='blue')
    plt.plot(years_test, test.values, ':', label='Actual', color='green')
    plt.fill_between(years_test, lower_bounds, upper_bounds, color='blue', alpha=0.4, label='90% Predictive Interval')
    plt.xlabel('Year')
    plt.ylabel(name)
    plt.title(f'Rolling Forecast for {name} with Exponential Smoothing + Residual Bootstrap')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return point_forecasts, np.array(all_samples)



def load_hmd_file(file_path, value_columns=['Female', 'Male', 'Total']):
    """Load and aggregate HMD file into yearly data indexed by datetime."""
    df = pd.read_csv(file_path, sep='\s+', comment='#', skiprows=2)
    df_yearly = df.groupby('Year')[value_columns].sum().round().astype(int).reset_index()
    df_yearly['Date'] = pd.to_datetime(df_yearly['Year'], format='%Y')
    df_yearly.set_index('Date', inplace=True)
    return df_yearly

def plot_series(series, title, ylabel):
    """Simple time series plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(series, linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():

    print('-----------------------------------------------------'
          '\n--------------Loading and Plotting Data--------------'
          '\n-----------------------------------------------------')

    # Load population and deaths
    yearly_population = load_hmd_file('Population.txt')
    yearly_deaths = load_hmd_file('Deaths_1x1.txt')

    # Plot total population and deaths
    plot_series(yearly_population['Total'], "Total Population Over Time (HMD)", "Population")
    plot_series(yearly_deaths['Total'], "Total Mortality Over Time (HMD)", "Deaths")

    # Compute and plot mortality rate
    mortality_rate = pd.DataFrame({
        'Total': yearly_deaths['Total'] / yearly_population['Total'],
        'Year': yearly_deaths['Year']
    })
    mortality_rate['Date'] = pd.to_datetime(mortality_rate['Year'], format='%Y')
    mortality_rate.set_index('Date', inplace=True)

    plot_series(mortality_rate['Total'], "Mortality Rate Over Time (HMD)", "Mortality Rate")

    print('-----------------------------------------------------'
          '\n---------Loading and Plotting Data Complete.---------'
          '\n-----------------------------------------------------')

    print('-----------------------------------------------------'
          '\n---------------Fitting Base Forecasts----------------'
          '\n-----------------------------------------------------')

    pop_base_fc = fit_base_forecast(yearly_population,name='Population')
    mort_base_fc = fit_base_forecast(yearly_deaths,name='Deaths')
    mort_rate_base_fc = fit_base_forecast(mortality_rate,name='Mortality Rate')

    pop_base_fc_rol = fit_base_forecast_rolling(yearly_population,name='Population')
    mort_base_fc_rol = fit_base_forecast(mortality_rate,name='Deaths')
    mort_rate_base_fc_rol = fit_base_forecast(mortality_rate,name='Mortality Rate')


    print(0)

if __name__ == "__main__":
    main()

