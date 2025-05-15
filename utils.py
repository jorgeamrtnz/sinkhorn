import numpy as np
from typing import Union

def construct_cost(
    x: np.ndarray, 
    y: Union[np.ndarray, None] = None, 
    p: float = 2
) -> np.ndarray:
    """
    Return M where M[i,j] = ||x[i] - y[j]||_p^p.

    Inputs
    ----------
    x : (n, d) array
    y : (m, d) array or None (defaults to x)
    p : order of the norm
    """

    # dimension check
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = x if y is None else np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]

    x = np.asarray(x)
    y = x if y is None else np.asarray(y)

    # M_ij = ||x_i - y_j||_2^2 = ||x||^2 + ||y||^2 + 2||x||||y||
    if p == 2:
        X2 = (x ** 2).sum(axis=1)[:, None]
        Y2 = (y ** 2).sum(axis=1)[None, :]
        M = X2 + Y2 - 2 * x @ y.T
        return np.maximum(M, 0.0)  # Ensure non-negativity

    # General case: M_ij = sum_k |x_ik - y_jk|^p
    diff = np.abs(x[:, None, :] - y[None, :, :]) ** p
    return diff.sum(axis=-1)