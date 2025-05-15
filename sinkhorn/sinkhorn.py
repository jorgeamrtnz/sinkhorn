import numpy as np
from utils import construct_cost

def sinkhorn_distance(M: np.ndarray, λ: float, r: np.ndarray, C: np.ndarray, *, max_iter: int=1000, δ: float=1e-9) -> np.ndarray:
    """
    Algorithm 1 from Cuturi (2013).

    Inputs
    ----------
    M : Ground-cost matrix.
    λ : Regularisation weight.
    r : Source histogram (must sum to 1).
    C : Each column is a target histogram c_j (must sum to 1).
    max_iter : Maximum Sinkhorn iterations.
    δ : Convergence tolerance on ‖u^{t+1} - u^{t}‖_∞.

    Computes
    -------
    d : Sinkhorn distances to each target.
    """
    # I = (r > 0); r = r(I); M = M(I, :); K = exp(−λM)
    I = r > 0
    r = r[I]
    M = M[I, :]
    K = np.exp(-λ * M)

    # u = ones(length(r), N)/length(r);
    if C.ndim == 1:
        C = C[:, None]
    N = C.shape[1]
    u = np.ones((r.shape[0], N), dtype=M.dtype) / r.size

    # K˜ = bsxfun(@rdivide, K, r) % equivalent to K˜ = diag(1./r)K
    K_ = K / r[:, None]

    # while u changes or any other relevant stopping criterion do
    for _ in range(max_iter):
        # TODO: Add numerical stability
        # u = 1./(K˜ (C./(K′u)))
        u_prev = u.copy()
        u = 1.0 / (K_ @ (C / (K.T @ u)))
        # end while
        if np.max(np.abs(u - u_prev)) < δ:
            break

    # v = C./(K′u)
    v = C / (K.T @ u)

    # d = sum(u. ∗ ((K. ∗ M)v)
    d = np.sum(u * ((K * M) @ v), axis=0)
    return d if d.size > 1 else d.item()


# toy example: two 1-D Gaussians
n = 50
x = np.linspace(-2, 2, n)
M = construct_cost(x, x, 1)     # p = 1 cost
r = np.exp(-x**2); r /= r.sum()
c = np.exp(-(x - 1)**2); c /= c.sum()

d = sinkhorn_distance(M, λ=10.0, r=r, C=c)
print(f"Sinkhorn distance ≈ {d:.4f}")