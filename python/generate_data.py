## --- Generate data for the bottom ---

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def generate_ar_processes(phi_1, phi_2, T=1000, case='independent', rho=0.95, scale=0.001, seed=42):
    np.random.seed(seed)

    b1 = np.zeros(T)
    b2 = np.zeros(T)

    if case == 'independent':
        eps_1 = np.random.normal(0, 0.1, T)
        eps_2 = np.random.normal(0, 0.1, T)
    elif case == 'correlated':
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        eps = scale * np.random.multivariate_normal(mean, cov, T)
        eps_1, eps_2 = eps[:, 0], eps[:, 1]
    else:
        raise ValueError("case must be either 'independent' or 'correlated'")

    for t in range(1, T):
        b1[t] = phi_1 * b1[t - 1] + eps_1[t]
        b2[t] = phi_2 * b2[t - 1] + eps_2[t]
    if case == 'correlated':
        b1 *= 5e2
        b2 *= 5e2
    u = b1 ** 2 + b2 ** 2


    # do some plots
    # lineplot of components
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    df = pd.DataFrame(np.vstack([b1, b2]).T, columns=['b1', 'b2'])
    df.plot(ax=ax[0])
    ax[1].scatter(df['b1'].values, df['b2'].values, s=1)
    plt.show()

    return pd.DataFrame({'B1': b1, 'B2': b2, 'U': u})


def main():
    phi_1 = 0.9
    phi_2 = 0.9
    T = 1000
    data_indep = generate_ar_processes(phi_1, phi_2, T=T, case='independent', seed=42)
    data_corr = generate_ar_processes(phi_1, phi_2, T=T, case='correlated', seed=42)
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