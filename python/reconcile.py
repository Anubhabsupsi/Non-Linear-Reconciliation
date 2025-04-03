import numpy as np
import pandas as pd
from bayesreconpy.reconc_BUIS import _check_weights, _resample, _distr_sample, _emp_pmf, _distr_pmf
import os
from typing import Union, Dict
from KDEpy import FFTKDE

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


def main():
    fc_folder = '../forecasts'
    test_data = pd.read_pickle(os.path.join(fc_folder, 'indep_test_dict.pkl'))

    f_quadratic = lambda B: B[:, 0]**2 + B[:, 1]**2
    ets_fc = pd.read_pickle(os.path.join(fc_folder, 'indep_base_fc_ets.pkl'))
    reconciled_ets_fc = {}
    for h in range(1, ets_fc['U'].shape[1] + 1):
        print(f"\nReconciliating horizon h={h}")

        sliced_dict = {
            key: ets_fc[key][:, h - 1, :]  # shape: (691, 1000)
            for key in ets_fc
        }

        df_dict = {
            key: pd.DataFrame(sliced_dict[key])
            for key in sliced_dict
        }

        reconciled = reconcile_rolling_forecasts_nonlinear(df_dict, relation=f_quadratic)

        # Store the reconciled result by horizon
        reconciled_ets_fc[f"h={h}"] = reconciled

    reconciled_ets_fc['U'].shape




if __name__ == "__main__":
    main()