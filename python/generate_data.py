## --- Generate data for the bottom ---

import numpy as np
import pandas as pd
import os

def generate_ar_processes(phi_1, phi_2, T=1000, case='independent', rho=0.8, seed=42):
    np.random.seed(seed)

    b1 = np.zeros(T)
    b2 = np.zeros(T)

    if case == 'independent':
        eps_1 = np.random.normal(0, 1, T)
        eps_2 = np.random.normal(0, 1, T)
    elif case == 'correlated':
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        eps = np.random.multivariate_normal(mean, cov, T)
        eps_1, eps_2 = eps[:, 0], eps[:, 1]
    else:
        raise ValueError("case must be either 'independent' or 'correlated'")

    for t in range(1, T):
        b1[t] = phi_1 * b1[t - 1] + eps_1[t]
        b2[t] = phi_2 * b2[t - 1] + eps_2[t]

    u = b1 ** 2 + b2 ** 2

    return pd.DataFrame({'B1': b1, 'B2': b2, 'U': u})

def main():
    phi_1 = np.random.uniform(0, 1, 1).item()
    phi_2 = np.random.uniform(0, 1, 1).item()
    T = 1000
    data_indep = generate_ar_processes(phi_1 = phi_1, phi_2 = phi_2, T=T, case='independent')
    data_corr = generate_ar_processes(phi_1 = phi_1, phi_2 = phi_2, T=T,case='correlated', rho=0.8)
    data_folder = '../data/'

    os.makedirs(data_folder, exist_ok=True)

    data_indep_file_name = os.path.join(data_folder, 'indep_ar_process.pkl')
    with open(data_indep_file_name, 'wb') as data_file:
        pd.to_pickle(data_indep, data_file)

    data_corr_file_name = os.path.join(data_folder, 'corr_ar_process.pkl')
    with open(data_corr_file_name, 'wb') as data_file:
        pd.to_pickle(data_corr, data_file)

if __name__ == '__main__':
    main()