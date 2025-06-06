import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import os
import pickle

def fit_predictive_model(data_indep, tr_ratio=0.8, plot_steps=10, n_estimators=100, criterion='absolute_error', n_samples=100):
    """
    Fits a predictive model using RandomForestRegressor (L1 loss) to forecast B1, B2, U
    based on lag-1 value, and visualizes predictions + predictive samples.

    Parameters:
    -----------
    data_indep : pd.DataFrame
        DataFrame with columns ['B1', 'B2', 'U'] (or more), indexed by time.
    tr_ratio : float
        Proportion of data to use for training (default 0.8).
    plot_steps : int
        Number of steps to plot in the prediction sample plot (default 10).
    n_estimators : int
        Number of estimators for RandomForest (default 100).
    criterion : str
        Loss function for RandomForest (default 'absolute_error').
    n_samples : int
        Desired number of predictive samples per step.

    Returns:
    --------
    y_hat_te_samples : np.ndarray
        Predictive samples tensor: shape (test_steps, n_samples, n_vars).
    y_hat_te : pd.DataFrame
        Point forecasts on test set.
    top_errors : pd.DataFrame
        Top 20% smallest L2-norm training residuals used for resampling.
    df_te : pd.DataFrame
        Test portion of the original input data.
    """
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    df_tr = data_indep.iloc[:int(tr_ratio * len(data_indep))]
    df_te = data_indep.iloc[int(tr_ratio * len(data_indep)):]

    # Initialize predictions
    y_hat_te = pd.DataFrame(index=df_te.index[:-1], columns=df_te.columns)
    y_hat_tr = pd.DataFrame(index=df_tr.index[:-1], columns=df_tr.columns)

    # Fit + predict for each column independently
    for col in df_tr.columns:
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
        model.fit(df_tr.iloc[:-1][[col]], df_tr.iloc[1:][col])
        y_hat_te[col] = model.predict(df_te.iloc[:-1][[col]])
        y_hat_tr[col] = model.predict(df_tr.iloc[:-1][[col]])

    # Plot predictions vs. ground truth

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Automatically assign consistent colors
    colors = plt.get_cmap("tab10")

    for i, col in enumerate(df_tr.columns):
        color = colors(i)

        # Ground truth: solid line
        ax.plot(df_te.index[:-1], df_te.iloc[:-1][col],
                label=f'{col} (Ground Truth)', color=color, linestyle='-')

        # Prediction: dashed line with same color
        ax.plot(df_te.index[:-1], y_hat_te[col],
                label=f'{col} (Prediction)', color=color, linestyle='--')

    ax.set_title('Predictions vs. Ground Truth')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # Compute training errors
    tr_errors = df_tr.iloc[1:] - y_hat_tr
    tr_errors.dropna(inplace=True)

    # Take 20% smallest errors (based on L2 norm)
    error_magnitudes = (tr_errors ** 2).sum(axis=1)
    top_errors = tr_errors.loc[error_magnitudes.nsmallest(int(len(error_magnitudes) * 0.2)).index]

    # Resample to match n_samples
    resampled_errors = top_errors.sample(n=n_samples, replace=True).values

    # Predictive samples tensor: shape (T_test, n_samples, D)
    y_hat_te_samples = y_hat_te.values[:, np.newaxis, :] + resampled_errors[np.newaxis, :, :]

    # Plot predictive samples (first N steps)
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.plot(df_te.iloc[:plot_steps].values, color='green', label='ground truth')
    plt.plot(y_hat_te.iloc[:plot_steps].values, color='red', label='prediction')
    xs = np.tile(np.arange(plot_steps), (y_hat_te_samples[:plot_steps, :, 0].shape[1], 1))
    for i in range(data_indep.shape[1]):  # Loop over variables
        plt.scatter(xs, np.squeeze(y_hat_te_samples[:plot_steps, :, i].T), alpha=0.6,
                    label=f'samples {data_indep.columns[i]}')
    plt.legend()
    plt.title('Predictive Samples (First Steps)')
    plt.show()

    # 3D plot using plotly (first 20 steps)
    fig = go.Figure()
    fig.update_layout(width=1000, height=800, title='3D Predictive Samples')
    for i in range(min(20, len(y_hat_te_samples))):
        fig.add_trace(
            go.Scatter3d(x=y_hat_te_samples[i, :, 0], y=y_hat_te_samples[i, :, 1], z=y_hat_te_samples[i, :, 2],
                         mode='markers', marker=dict(size=3, color='blue', opacity=0.6)))
        fig.add_trace(go.Scatter3d(x=[y_hat_te.iloc[i, 0]], y=[y_hat_te.iloc[i, 1]], z=[y_hat_te.iloc[i, 2]],
                                   mode='markers', marker=dict(size=6, color='red')))
        fig.add_trace(go.Scatter3d(x=[df_te.iloc[i, 0]], y=[df_te.iloc[i, 1]], z=[df_te.iloc[i, 2]],
                                   mode='markers', marker=dict(size=6, color='green')))

    # Plot paraboloid surface
    x = np.linspace(-0.6, 0.6, 100)
    y = np.linspace(-0.6, 0.6, 100)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.3, colorscale='Blues'))
    fig.show()

    # Distribution plots of B1, B2, U in training set
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    df_tr.plot.hist(ax=ax[0], bins=100, alpha=0.5)
    df_tr['B1'].plot.kde(ax=ax[0], color='red')
    df_tr['B2'].plot.kde(ax=ax[1], color='red')
    df_tr['U'].plot.kde(ax=ax[2], color='red')
    ax[0].set_title('B1 Distribution')
    ax[1].set_title('B2 Distribution')
    ax[2].set_title('U Distribution')
    plt.show()

    return y_hat_te_samples, y_hat_te, top_errors, df_te


def main():
    data_folder = '../data/'
    fc_folder = '../forecasts/'
    os.makedirs(fc_folder, exist_ok=True)

    # Load data
    data_indep = pd.read_pickle(os.path.join(data_folder, 'indep_ar_process.pkl'))
    data_corr = pd.read_pickle(os.path.join(data_folder, 'corr_ar_process.pkl'))

    # Independent data
    n_samples = 500
    base_fc_indep, indep_det_fc, indep_tr_errors, df_te = fit_predictive_model(data_indep,n_samples=n_samples)
    save_path = os.path.join(fc_folder, f'base_fc_indep_{n_samples}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(base_fc_indep, f)
    res_path = os.path.join(fc_folder, f'res_indep_{n_samples}.pkl')
    with open(res_path, 'wb') as f:
        pickle.dump(indep_tr_errors, f)
    det_path = os.path.join(fc_folder, f'det_fc_indep_{n_samples}.pkl')
    with open(det_path, 'wb') as f:
        pickle.dump(indep_det_fc, f)
    df_path = os.path.join(fc_folder, f'indep_te_{n_samples}.pkl')
    with open(df_path, 'wb') as f:
        pickle.dump(df_te, f)

    # Correlated data
    base_fc_corr, corr_det_fc, corr_tr_errors, df_corr_te = fit_predictive_model(data_corr,n_samples=n_samples)
    save_path = os.path.join(fc_folder, f'base_fc_corr_{n_samples}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(base_fc_corr, f)
    res_path = os.path.join(fc_folder, f'res_corr_{n_samples}.pkl')
    with open(res_path, 'wb') as f:
        pickle.dump(corr_tr_errors, f)
    det_path = os.path.join(fc_folder, f'det_fc_corr_{n_samples}.pkl')
    with open(det_path, 'wb') as f:
        pickle.dump(corr_det_fc, f)
    df_path = os.path.join(fc_folder, f'corr_te_{n_samples}.pkl')
    with open(df_path, 'wb') as f:
        pickle.dump(df_corr_te, f)


if __name__ == '__main__':
    main()