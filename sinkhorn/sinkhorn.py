import numpy as np
from . import utils
from scipy.special import logsumexp

# def sinkhorn_distance(M: np.ndarray, λ: float, r: np.ndarray, C: np.ndarray, *, max_iter: int=1000, δ: float=1e-9) -> np.ndarray:
#     """
#     Algorithm 1 from Cuturi (2013).

#     Inputs
#     ----------
#     M : Ground-cost matrix.
#     λ : Regularisation weight.
#     r : Source histogram (must sum to 1).
#     C : Each column is a target histogram c_j (must sum to 1).
#     max_iter : Maximum Sinkhorn iterations.
#     δ : Convergence tolerance on ‖u^{t+1} - u^{t}‖_∞.

#     Computes
#     -------
#     d : Sinkhorn distances to each target.
#     """
#     # I = (r > 0); r = r(I); M = M(I, :); K = exp(−λM)
#     I = r > 0
#     r = r[I]
#     M = M[I, :]
#     K = np.exp(-λ * M)

#     # u = ones(length(r), N)/length(r);
#     if C.ndim == 1:
#         C = C[:, None]
#     N = C.shape[1]
#     u = np.ones((r.shape[0], N), dtype=M.dtype) / r.size

#     # K˜ = bsxfun(@rdivide, K, r) % equivalent to K˜ = diag(1./r)K
#     K_ = K / r[:, None]

#     # while u changes or any other relevant stopping criterion do
#     for _ in range(max_iter):
#         # TODO: Add numerical stability
#         # u = 1./(K˜ (C./(K′u)))
#         u_prev = u.copy()
#         u = 1.0 / (K_ @ (C / (K.T @ u)))
#         # end while
#         if np.max(np.abs(u - u_prev)) < δ:
#             break

#     # v = C./(K′u)
#     v = C / (K.T @ u)

#     # d = sum(u. ∗ ((K. ∗ M)v)
#     d = np.sum(u * ((K * M) @ v), axis=0)
#     return d if d.size > 1 else d.item()



def sinkhorn_distance(M: np.ndarray, λ: float, r: np.ndarray, C: np.ndarray, *, max_iter: int=1000, δ: float=1e-9) -> np.ndarray:
    """
    Explicit version, more aligned with the proposal for stability
    """
    eps = 1.0 / λ    # ε in the paper
    n, m = M.shape

    log_r = np.log(r + 1e-300)
    log_C = np.log(C + 1e-300)

    # initialise dual potentials μ, v  (log-domain)
    μ = np.zeros(n)
    v = np.zeros(m)

    for _ in range(max_iter):
        μ_prev = μ.copy()

        # μ-update   μ_i ← ε[ log r_i − log Σ_j exp((−M_ij + v_j)/ε) ]
        μ = eps * (log_r - logsumexp((-M + v[None, :]) / eps, axis=1))

        # -update   v_j ← ε[ log C_j − log Σ_i exp((−M_ij + μ_i)/ε) ]
        v = eps * (log_C - logsumexp((-M + μ[:, None]) / eps, axis=0))

        if np.max(np.abs(μ - μ_prev)) < δ:
            break

    # transport plan   P_ij = exp((μ_i + v_j − M_ij)/ε)
    P = np.exp((μ[:, None] + v[None, :] - M) / eps)

    return np.sum(P * M)
