import numpy as np

def compute_crps(y_section, y_hat_section, q_min=None):
    start_from = 0 if q_min is None else int(y_hat_section.shape[1] * q_min)
    y_hat_sorted = np.sort(y_hat_section[:, start_from:], axis=1)
    m = y_hat_sorted.shape[1]
    return (2 / m) * np.nanmean((y_hat_sorted - y_section[:, None]) *
                                (m * (y_section[:, None] < y_hat_sorted) - np.arange(1, m + 1) + 0.5),
                                axis=1)