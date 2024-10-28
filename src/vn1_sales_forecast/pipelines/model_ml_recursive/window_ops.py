import operator as op

import numpy as np
from mlforecast.lag_transforms import Combine, Offset, RollingMean
from numba import njit


def Trend(w):
    return Combine(RollingMean(1), Offset(RollingMean(w), 1), op.truediv)


def Momentum(w):
    return Combine(RollingMean(w), Offset(RollingMean(w), w), op.truediv)


@njit
def rolling_zero_shares(x, window_size: int = 7) -> np.ndarray:
    n_samples = len(x)
    result = np.full(n_samples, np.nan)

    if n_samples < window_size:
        return result

    for i in range(window_size, n_samples):
        xs = x[i - window_size + 1 : i + 1]
        result[i] = np.sum(xs == 0) / window_size
    return result


@njit
def rolling_slope(x: np.ndarray, window_size: int = 7) -> np.ndarray:
    n_samples = len(x)
    result = np.full(n_samples, np.nan)

    if n_samples < window_size:
        return result

    for i in range(window_size, n_samples):
        xs = x[i - window_size + 1 : i + 1]

        m = window_size - 1
        xs_range_mean = m / 2
        xs_range = np.arange(0, window_size)
        xs_mean = xs.mean()
        beta = np.dot(xs - xs_mean, xs_range - xs_range_mean) / (m * np.var(xs_range))
        alpha = xs_mean - beta * xs_range_mean
        result[i] = alpha
    return result
