import numpy as np
import pandas as pd
from bayesreconpy.reconc_BUIS import _check_weights, _resample, _distr_sample, _emp_pmf, _distr_pmf
import os
from typing import Union, Dict
from KDEpy import FFTKDE
import jax
from tqdm import tqdm
import gc
from jax import vmap
import jax.numpy as jnp
from bayesreconpy.shrink_cov import _schafer_strimmer_cov as schafer_strimmer_cov
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
from scipy.special import gamma
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from matplotlib import MatplotlibDeprecationWarning
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message="The converter attribute was deprecated")

def _compute_weights(b, u, in_type_, distr_):
    if in_type_ == "samples":
        if distr_ == "discrete":
            w = _emp_pmf(b, u)
        elif distr_ == "continuous":
            # Use fixed bw if data too small
            try:
                fftkde = FFTKDE(kernel='gaussian', bw='ISJ').fit(u)
            except ValueError:
                fftkde = FFTKDE(kernel='gaussian', bw=0.1).fit(u)

            grid_lims = [np.minimum(np.min(u), np.min(b)), np.maximum(np.max(u), np.max(b))]
            grid = np.linspace(grid_lims[0] - 1e-3, grid_lims[1] + 1e-3, 1000)
            res = fftkde.evaluate(grid)
            w = np.interp(b, grid, res)

        w[np.isnan(w)] = 0
    elif in_type_ == "params":
        w = _distr_pmf(b, u, distr_)

    if np.sum(w) == 0:
        w = np.ones_like(w)

    return w

def reconc_BUIS_nonlinear(
    U_input: Union[np.ndarray, Dict[str, float]],
    B_samples: np.ndarray,
    f_B_to_U: callable,
    in_type: str = "samples",
    distr: str = "continuous",
    num_samples: int = 20000,
    suppress_warnings: bool = False,
    seed: Union[int, None] = None
) -> dict:
    """
    Non-linear BUIS reconciliation for U = f(B), where U and B are forecasts.
    Supports sample-based and param-based upper-level inputs.

    Parameters
    ----------
    U_input : np.ndarray or dict
        Upper-level forecast samples (if in_type="samples") or distribution parameters (if in_type="params").

    B_samples : np.ndarray, shape (N, n_bottom)
        Bottom-level forecast samples. Each row is one joint draw of all bottom nodes.

    f_B_to_U : callable
        Function that computes U = f(B). Must accept shape (N, n_bottom) and return (N,).

    in_type : str
        Either 'samples' or 'params'.

    distr : str
        'continuous' if using samples, or one of {'gaussian', 'poisson', 'nbinom'} if using params.

    num_samples : int
        Used only when in_type='params'.

    suppress_warnings : bool
        Suppress diagnostic messages.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
            'bottom_reconciled_samples': (n_bottom, N),
            'upper_reconciled_samples': (1, N),
            'reconciled_samples': (n_bottom + 1, N)
        }
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Sample U if needed
    if in_type == "params":
        U_samples = _distr_sample(U_input, distr, num_samples)
    elif in_type == "samples":
        U_samples = U_input
    else:
        raise ValueError(f"Invalid in_type: {in_type}. Must be 'samples' or 'params'.")

    # Step 2: Match sample counts
    if B_samples.shape[0] != U_samples.shape[0]:
        raise ValueError(f"Sample size mismatch: B has {B_samples.shape[0]} rows, U has {U_samples.shape[0]}")

    # Step 3: Compute U = f(B)
    U_from_B = f_B_to_U(B_samples)

    # Step 4: Importance weights
    weights = _compute_weights(
        b=U_from_B,
        u=U_samples,
        in_type_=in_type,
        distr_=distr
    )

    check = _check_weights(weights)
    if check["warning"] and not suppress_warnings:
        for msg in check["warning_msg"]:
            print(f"⚠️ Warning: {msg}")

    # Step 5: Resample
    B_resampled = _resample(B_samples, weights)

    # Step 6: Recompute upper values using f(B)
    U_reconc = f_B_to_U(B_resampled)

    return {
        "bottom_reconciled_samples": B_resampled.T,
        "upper_reconciled_samples": U_reconc.reshape(1, -1),
        "reconciled_samples": np.vstack([U_reconc, B_resampled.T])
    }

def reconcile_rolling_forecasts_nonlinear(
    fc_dict: Dict[str, pd.DataFrame],
    relation: callable
) -> Dict[str, pd.DataFrame]:

    b1_df, b2_df, u_df = fc_dict['B1'], fc_dict['B2'], fc_dict['U']
    h_cols = b1_df.columns
    test_dates = b1_df.index

    # Create empty results
    recon_b1 = pd.DataFrame(index=test_dates, columns=h_cols)
    recon_b2 = pd.DataFrame(index=test_dates, columns=h_cols)
    recon_u = pd.DataFrame(index=test_dates, columns=h_cols)

    for dt in test_dates:
        B1_row = b1_df.loc[dt].values
        B2_row = b2_df.loc[dt].values
        U_row = u_df.loc[dt].values

        # Stack bottom samples: shape (H, 2)
        B = np.column_stack([B1_row, B2_row])

        result = reconc_BUIS_nonlinear(
            U_input=U_row,
            B_samples=B,
            f_B_to_U=relation,
            in_type="samples",
            distr="continuous"
        )

        recon_b1.loc[dt] = result['bottom_reconciled_samples'][0]
        recon_b2.loc[dt] = result['bottom_reconciled_samples'][1]
        recon_u.loc[dt] = result['upper_reconciled_samples'][0]

    return {
        'B1': recon_b1.astype(float),
        'B2': recon_b2.astype(float),
        'U': recon_u.astype(float)
    }

def unscented_transform(mu_x, Sigma_x, h, R, alpha=1e-3, beta=2, kappa=0):
    n = mu_x.shape[0]
    lam = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lam)

    # Weights
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    Wc = np.copy(Wm)
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    # Sigma points
    S = np.linalg.cholesky(Sigma_x)
    sigma_pts = np.zeros((2 * n + 1, n))
    sigma_pts[0] = mu_x
    for i in range(n):
        sigma_pts[i + 1] = mu_x + gamma * S[:, i]
        sigma_pts[n + i + 1] = mu_x - gamma * S[:, i]

    # Transform through h(.)
    z_sigma = np.array([h(pt) for pt in sigma_pts])
    m = z_sigma.shape[1]

    # Predicted measurement mean
    y_pred = np.sum(Wm[:, None] * z_sigma, axis=0)

    # Measurement covariance and cross covariance
    S_y = R.copy()
    P_xy = np.zeros((n, m))
    for i in range(2 * n + 1):
        dz = z_sigma[i] - y_pred
        dx = sigma_pts[i] - mu_x
        S_y += Wc[i] * np.outer(dz, dz)
        P_xy += Wc[i] * np.outer(dx, dz)

    # Kalman gain
    K = P_xy @ np.linalg.inv(S_y)

    return lambda y_obs: (
        mu_x + K @ (y_obs - y_pred),
        Sigma_x - K @ S_y @ K.T
    )

def sample_multivariate_gaussian(mean, cov, n_samples=1, random_state=None):
    """
    Draw samples from a multivariate normal distribution.

    Parameters:
    - mean: array-like, shape (d,)
    - cov: array-like, shape (d, d)
    - n_samples: int, number of samples to draw
    - random_state: int or np.random.Generator, optional

    Returns:
    - samples: np.ndarray, shape (n_samples, d)
    """
    rng = np.random.default_rng(random_state)
    return rng.multivariate_normal(mean, cov, size=n_samples)


def h(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.array([x[0]**2 + x[1]**2])
    elif x.ndim == 2:
        return (x[:, 0]**2 + x[:, 1]**2)[:, None]
    elif x.ndim == 3:
        return (x[..., 0]**2 + x[..., 1]**2)[..., None]
    else:
        raise ValueError("Input should be (n,), (n_samples, n), or (n_points, n_samples, n)")

def generalized_energy_score(y_true: np.ndarray, y_samples: np.ndarray, alpha: float = 1.0, batch_size: int = 1000) -> np.ndarray:
    """
    Compute the generalized energy score (with alpha parameter) efficiently.

    Parameters
    ----------
    y_true : array of shape (n, d)
        Observed true values.
    y_samples : array of shape (n, m, d)
        Forecast samples. m = number of samples.
    alpha : float, optional (default=1.0)
        Exponent in the generalized energy score (must be in (0, 2]).
    batch_size : int
        Number of sample pairs to compute per batch to reduce memory load.

    Returns
    -------
    energy_scores : array of shape (n,)
        The generalized energy score for each observation.
    """
    assert 0 < alpha <= 2, "alpha must be in (0, 2]"

    n, m, d = y_samples.shape
    y_true = y_true[:, np.newaxis, :]  # (n, 1, d)

    # First term: E[||y - Y||^alpha]
    term1 = np.linalg.norm(y_samples - y_true, axis=2) ** alpha
    term1 = term1.mean(axis=1)

    # Second term: E[||Y - Y'||^alpha] using batching
    term2 = np.zeros(n)
    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        batch = y_samples[:, i:end, :]  # shape: (n, batch, d)
        for j in range(0, m, batch_size):
            end_j = min(j + batch_size, m)
            batch_j = y_samples[:, j:end_j, :]  # shape: (n, batch_j, d)

            # shape: (n, batch, batch_j)
            dists = np.linalg.norm(batch[:, :, np.newaxis, :] - batch_j[:, np.newaxis, :, :], axis=3) ** alpha
            term2 += dists.mean(axis=(1, 2))

    term2 *= 0.5 * (batch_size ** 2) / (m ** 2)  # scale to match full expectation
    return term1 - term2

def plot_paraboloid_results(indexes, y_hat_te, y_hat_te_samples, y_rec_te_samples, df_te, title_suffix=""):
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    fig.update_layout(width=1000, height=800, title=f"Top cases: {title_suffix}")
    for idx in indexes:
        y_hat_te_max = y_hat_te.iloc[idx].values
        y_hat_te_samples_max = y_hat_te_samples[idx]
        y_rec_te_samples_max = y_rec_te_samples[idx]
        y_true_max = df_te.iloc[idx].values

        fig.add_trace(go.Scatter3d(x=y_hat_te_samples_max[:, 0], y=y_hat_te_samples_max[:, 1], z=y_hat_te_samples_max[:, 2],
                                   mode='markers', marker=dict(size=3, color='blue', opacity=0.6),
                                   name='Baseline Samples'))
        fig.add_trace(go.Scatter3d(x=y_rec_te_samples_max[:, 0], y=y_rec_te_samples_max[:, 1], z=y_rec_te_samples_max[:, 2],
                                   mode='markers', marker=dict(size=3, color='orange', opacity=0.6),
                                   name='Reconciled Samples'))
        fig.add_trace(go.Scatter3d(x=[y_hat_te_max[0]], y=[y_hat_te_max[1]], z=[y_hat_te_max[2]],
                                   mode='markers', marker=dict(size=6, color='red'), name='Point Forecast'))
        fig.add_trace(go.Scatter3d(x=[y_true_max[0]], y=[y_true_max[1]], z=[y_true_max[2]],
                                   mode='markers', marker=dict(size=6, color='green'), name='True Value'))

    x = np.linspace(-0.6, 0.6, 100)
    y = np.linspace(-0.6, 0.6, 100)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 + Y ** 2
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.3, colorscale='Blues'))
    fig.show()

def analyze_reconciliation(y_hat_te, y_hat_te_samples, y_rec_te_samples, df_te, batch_size=1000, method_name="",
                            bandwidth=0.2, kernel="epanechnikov", nu=3):
    energy_score_baseline = generalized_energy_score(df_te.iloc[:-1].values, y_hat_te_samples, batch_size=batch_size)
    energy_score_rec = generalized_energy_score(df_te.iloc[:-1].values, y_rec_te_samples, batch_size=batch_size)
    es_diff = energy_score_baseline - energy_score_rec

    log_score_baseline = kde_log_scores_all(df_te.iloc[:-1, :2].values, y_hat_te_samples[:, :, :2], bandwidth=bandwidth,
                                            kernel=kernel, nu=nu)
    log_score_rec = kde_log_scores_all(df_te.iloc[:-1, :2].values, y_rec_te_samples[:, :, :2], bandwidth=bandwidth,
                                       kernel=kernel, nu=nu)
    ls_diff = log_score_baseline - log_score_rec

    improvement_index = np.argsort(es_diff)
    print(f'[{method_name}] Energy score improved for {(es_diff > 0).sum()} out of {len(es_diff)} points')
    print(f'Average ES: {np.mean(energy_score_rec)} for {method_name}')
    print(f'Average baseline ES: {np.mean(energy_score_baseline)}')
    print(f'Average Log Score: {np.mean(log_score_rec)} for {method_name}')
    print(f'Average baseline Log Score: {np.mean(log_score_baseline)}')
    print(f'Log score improved in {(ls_diff > 0).sum()} out of {len(ls_diff)} points')

    # Plotting top and bottom cases
    best_title = f"{method_name} - Best 3\nES diff: {es_diff[improvement_index[-3:]]}\nLS diff: {ls_diff[improvement_index[-3:]]}"
    worst_title = f"{method_name} - Worst 3\nES diff: {es_diff[improvement_index[:3]]}\nLS diff: {ls_diff[improvement_index[:3]]}"
    plot_paraboloid_results(improvement_index[-3:], y_hat_te, y_hat_te_samples, y_rec_te_samples, df_te, best_title)
    plot_paraboloid_results(improvement_index[:3], y_hat_te, y_hat_te_samples, y_rec_te_samples, df_te, worst_title)

    inside = lambda x: x[:, :, 0] ** 2 + x[:, :, 1] ** 2 < x[:, :, 2]
    samples_inside_avg = inside(y_hat_te_samples).mean(axis=1)
    point_forecast_inside = inside(y_hat_te.values[:, np.newaxis, :]).mean(axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Top-left: Energy score vs samples
    ax[0, 0].scatter(samples_inside_avg, es_diff, s=10, alpha=0.7, color='blue')
    ax[0, 0].plot(np.linspace(0, 1, 100), np.zeros(100), 'k--')
    ax[0, 0].set_xlabel('Avg baseline samples inside paraboloid')
    ax[0, 0].set_ylabel('Energy Score diff')
    ax[0, 0].set_title(f'{method_name} - ES vs Samples')

    # Top-right: Log score vs samples
    ax[0, 1].scatter(samples_inside_avg, ls_diff, s=10, alpha=0.7, color='green')
    ax[0, 1].plot(np.linspace(0, 1, 100), np.zeros(100), 'k--')
    ax[0, 1].set_xlabel('Avg baseline samples inside paraboloid')
    ax[0, 1].set_ylabel('Log Score diff')
    ax[0, 1].set_title(f'{method_name} - LS vs Samples')

    # Bottom-left: Energy score vs point forecast
    ax[1, 0].scatter(point_forecast_inside, es_diff, s=10, alpha=0.7, color='blue')
    ax[1, 0].plot(np.linspace(0, 1, 100), np.zeros(100), 'k--')
    ax[1, 0].set_xlabel('Point forecast inside paraboloid')
    ax[1, 0].set_ylabel('Energy Score diff')
    ax[1, 0].set_title(f'{method_name} - ES vs Point Forecast')

    # Bottom-right: Log score vs point forecast
    ax[1, 1].scatter(point_forecast_inside, ls_diff, s=10, alpha=0.7, color='green')
    ax[1, 1].plot(np.linspace(0, 1, 100), np.zeros(100), 'k--')
    ax[1, 1].set_xlabel('Point forecast inside paraboloid')
    ax[1, 1].set_ylabel('Log Score diff')
    ax[1, 1].set_title(f'{method_name} - LS vs Point Forecast')

    plt.suptitle(f'{method_name} - Score Diagnostics', fontsize=16)
    plt.show()



import numpy as np
from scipy.spatial.distance import cdist

def kde_log_scores_all(y_true: np.ndarray, y_samples: np.ndarray, bandwidth=0.2, kernel='gaussian', nu=3) -> np.ndarray:
    """
    Compute KDE-based log scores for all test points in batch.

    Parameters
    ----------
    y_true : array of shape (n, d)
        True observations.
    y_samples : array of shape (n, m, d)
        Forecast samples per observation.
    bandwidth : float
        Bandwidth for the kernel.
    kernel : str
        'gaussian', 'epanechnikov', 't', or 'exponential'
    nu : float
        Degrees of freedom for t kernel (default=3).

    Returns
    -------
    log_scores : array of shape (n,)
        KDE log scores per observation.
    """
    n, m, d = y_samples.shape
    log_scores = np.zeros(n)

    for i in range(n):
        yt = y_true[i][None, :]
        ys = y_samples[i]
        distances = cdist(yt, ys)[0] / bandwidth

        if kernel == 'gaussian':
            kernel_vals = np.exp(-0.5 * distances**2) / (np.sqrt(2 * np.pi) ** d)
        elif kernel == 'epanechnikov':
            kernel_vals = 0.75 * (1 - distances**2)
            kernel_vals[distances > 1] = 0
        elif kernel == 't':
            if nu <= 0:
                raise ValueError("Degrees of freedom nu must be positive.")
            coef = gamma((nu + d) / 2) / (gamma(nu / 2) * (np.pi * nu) ** (d / 2))
            kernel_vals = coef * (1 + (distances ** 2) / nu) ** (-(nu + d) / 2)
        elif kernel == 'exponential':
            # exponential kernel (Laplacian-like, only positive side)
            kernel_vals = np.exp(-distances)
        else:
            raise ValueError("Unsupported kernel. Choose 'gaussian', 'epanechnikov', 't', or 'exponential'.")

        density_estimate = np.sum(kernel_vals) / (m * (bandwidth ** d))
        log_scores[i] = -np.log(density_estimate + 1e-12)

    return log_scores


def main():
    fc_folder = '../forecasts'
    n_samples = 500
    indep_base_fc = pd.read_pickle(os.path.join(fc_folder, f'base_fc_indep_{n_samples}.pkl'))
    indep_tr_res = pd.read_pickle(os.path.join(fc_folder, f'res_indep_{n_samples}.pkl'))
    y_hat_indep = pd.read_pickle(os.path.join(fc_folder, f'det_fc_indep_{n_samples}.pkl'))
    df_te = pd.read_pickle(os.path.join(fc_folder, f'indep_te_{n_samples}.pkl'))

    ## ---- Probabilistic Bottom-Up ----

    print('----------------------------------------------------------')
    print('----------------Running Probabilistic BU------------------')
    print('----------------------------------------------------------')

    bot_base_indep = indep_base_fc[:, :, :2]
    upp_bu = h(bot_base_indep)
    bu_indep_fc = np.concatenate((bot_base_indep,upp_bu),axis=2)

    print('Probabilistic BU completed ✅')

    print("BU fc shape:", bu_indep_fc.shape)

    ## ---- BUIS ----

    print('----------------------------------------------------------')
    print('---------------------Running BUIS-------------------------')
    print('----------------------------------------------------------')

    results = []
    f_quadratic = lambda B: B[:, 0] ** 2 + B[:, 1] ** 2
    for i in range(indep_base_fc.shape[0]):
        results.append(reconc_BUIS_nonlinear(
            U_input=indep_base_fc[i, :, 2],
            B_samples=indep_base_fc[i, :, :2],
            f_B_to_U=f_quadratic,
            in_type="samples",
            distr="continuous",
            suppress_warnings=True
        ))

    buis_indep_fc = np.stack([np.vstack([r['bottom_reconciled_samples'], r['upper_reconciled_samples']]).T for r in results], axis=0)

    print('BUIS completed ✅')

    print(buis_indep_fc.shape)

    ## ---- Orthogonal Projection ----

    print('----------------------------------------------------------')
    print('---------------Running Orthogonal Projection--------------')
    print('----------------------------------------------------------')

    # Define the paraboloid function f(y, x1, x2) = y - (x1^2 + x2^2)
    def f(Z):
        x1, x2, y = Z
        return y - (x1 ** 2 + x2 ** 2)

    # Jacobian of f using JAX
    J = jax.grad(f)

    import jax.numpy as jnp
    from jax import jacfwd
    from functools import partial

    def project_to_manifold(Z_hat, f, max_iter=100, tol=1e-6):
        Z = jnp.array(Z_hat)
        m = f(Z).size
        lambda_ = jnp.zeros((m, 1))  # Make it a column vector

        for i in range(max_iter):
            f_val = f(Z).reshape((m, 1))   # shape (m,1)
            J_f = jacfwd(f)(Z).reshape((m, -1))  # shape (m, 3)

            grad_L = Z - Z_hat + (J_f.T @ lambda_).reshape(-1)

            res = jnp.concatenate([grad_L, f_val.reshape(-1)], axis=0)

            n = Z.size
            KKT_matrix = jnp.block([
                [jnp.eye(n), J_f.T],
                [J_f, jnp.zeros((m, m))]
            ])

            delta = jnp.linalg.solve(KKT_matrix, -res)
            dZ, dlambda = delta[:n], delta[n:]

            Z = Z + dZ
            lambda_ = lambda_ + dlambda.reshape(-1, 1)

        return Z
    #project_all = vmap(vmap(project_to_manifold, in_axes=(0,)), in_axes=(0,))
    project_fn = partial(project_to_manifold, f=f)
    project_all = vmap(vmap(project_fn, in_axes=0), in_axes=0)
    proj_indep_fc = project_all(indep_base_fc)
    #project_all = vmap(vmap(project_fn, in_axes=0), in_axes=0)

    print('Orthogonal Projection completed ✅')
    print(proj_indep_fc.shape)

    ## ---- Unscented Kalman Filter ----

    print('----------------------------------------------------------')
    print('----------------Running Unscented Kalman------------------')
    print('----------------------------------------------------------')

    cov_res = schafer_strimmer_cov(indep_tr_res)['shrink_cov']
    mu_post = np.zeros((indep_base_fc.shape[0], 2))
    Sigma_post = np.zeros((indep_base_fc.shape[0], 2, 2))
    for i in range(indep_base_fc.shape[0]):
        mu_x = y_hat_indep.iloc[i].values[:2]
        sig_x = cov_res[:2, :2]
        R = cov_res[2, 2]

        ukf_update = unscented_transform(mu_x, sig_x, h, R)
        y = y_hat_indep.iloc[i].values[2]

        mu_post[i, :], Sigma_post[i, :, :] = ukf_update(y)

    bot_sample = np.zeros((indep_base_fc.shape[0], indep_base_fc.shape[1], 2))
    for i in range(indep_base_fc.shape[0]):
        bot_sample[i, :, :] = sample_multivariate_gaussian(mu_post[i, :], Sigma_post[i, :, :],
                                                           n_samples=indep_base_fc.shape[1])

    up_sample = h(bot_sample)
    ukf_indep_fc = np.concatenate((bot_sample, up_sample), axis=2)

    print('UKF completed ✅')
    print(ukf_indep_fc.shape)

    # === Plotting phase ===
    batch_size_1 = 500
    # Plot Probabilistic BU
    # After BU
    analyze_reconciliation(y_hat_indep, indep_base_fc, bu_indep_fc, df_te, batch_size=batch_size_1, method_name="BU")
    del bu_indep_fc
    gc.collect()

    # After BUIS
    analyze_reconciliation(y_hat_indep, indep_base_fc, buis_indep_fc, df_te, batch_size=batch_size_1,
                           method_name="BUIS")
    del buis_indep_fc
    gc.collect()

    # After Projection
    analyze_reconciliation(y_hat_indep, indep_base_fc, proj_indep_fc, df_te, batch_size=batch_size_1,
                           method_name="Projection")
    del proj_indep_fc
    gc.collect()

    # After UKF
    analyze_reconciliation(y_hat_indep, indep_base_fc, ukf_indep_fc, df_te, batch_size=batch_size_1, method_name="UKF")
    del ukf_indep_fc
    gc.collect()
    print(0)





if __name__ == '__main__':
    main()

