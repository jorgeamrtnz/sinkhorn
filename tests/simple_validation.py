# tests/test_validation.py
import numpy as np
import scipy.optimize as opt
import pytest

from sinkhorn.sinkhorn import sinkhorn_distance
from sinkhorn.utils   import construct_cost


# ---------- helpers ---------------------------------------------------------
def exact_ot_cost(M, a, b):
    """Solve the exact LP (Monge-Kantorovich) with SciPy linprog."""
    n, m = M.shape
    # Flatten variables: P_{ij} becomes z[k] with k = i*m + j
    c = M.ravel()
    # Constraints: (row sums) + (col sums) == 1 and positivity
    # A_eq z = rhs
    A = []
    rhs = []
    # Row constraints
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1.0
        A.append(row); rhs.append(a[i])
    # Col constraints
    for j in range(m):
        col = np.zeros(n * m)
        col[j::m] = 1.0
        A.append(col); rhs.append(b[j])

    res = opt.linprog(c, A_eq=A, b_eq=rhs, bounds=(0, None), method="highs")
    assert res.success, "LP failed!"
    return res.fun


def check_marginals(u, v, K, a, b, atol=1e-6):
    P = (u[:, None] * K) * v[None, :]
    assert np.allclose(P.sum(axis=1), a, atol=atol)
    assert np.allclose(P.sum(axis=0), b, atol=atol)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n, p, lam, rtol", [
    (10, 1, 50,  5e-2),   # Manhattan, λ moderate → 5 % tolerance
    (10, 2, 100, 2e-2),   # squared Eucl.
    (20, 2, 200, 3e-2),   # bigger grid requires larger λ
])
def test_sinkhorn_vs_lp(n, p, lam, rtol):
    """Numerical convergence: Sinkhorn → exact OT as λ→∞."""

    # 1-D grid
    x = np.linspace(-1, 1, n)[:, None]        # (n,1)
    y = np.linspace(-.8, .8, n)[:, None]

    M = construct_cost(x, y, p=p)
    a = np.exp(-x.squeeze()**2); a /= a.sum()
    b = np.exp(-(y.squeeze()-0.3)**2); b /= b.sum()

    # exact cost
    C_exact = exact_ot_cost(M, a, b)

    # Sinkhorn
    C_sink = sinkhorn_distance(M=M, λ=lam, r=a, C=b)

    # should be within ~1% for λ ≥ 100
    assert np.isclose(C_sink, C_exact, rtol=rtol)


def test_closed_form_two_point():
    """2-point measures with analytic OT."""
    x = np.array([[0.0], [1.0]])
    y = np.array([[0.0], [1.0]])
    a = np.array([0.3, 0.7])
    b = np.array([0.6, 0.4])
    M = construct_cost(x, y, p=1)         # cost = |x-y|

    # closed form: optimal plan ships min masses across identical points
    C_exact = 0.0 * min(a[0], b[0]) + 1.0 * abs(a[0]-b[0])

    C_sink = sinkhorn_distance(M, λ=20.0, r=a, C=b)
    assert np.isclose(C_sink, C_exact, atol=5e-3)
