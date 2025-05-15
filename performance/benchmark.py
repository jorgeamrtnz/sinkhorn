"""
Benchmark Sinkhorn vs LP (and optionally POT's Greenkhorn).

Usage
-----
python -m performance.benchmark \
    --algos Sinkhorn,LP,Greenkhorn --outfile bench.csv
"""
import time, csv, argparse, itertools, importlib
import numpy as np
from sinkhorn.sinkhorn   import sinkhorn_distance
from sinkhorn.utils      import construct_cost
from tests.simple_validation import exact_ot_cost  # now importable after __init__.py

# ---------- FLOP estimate ---------------------------------------------------
def flops_sinkhorn(n, m, iters=100):      # 100 ≈ default convergence
    return 5 * n * m * iters              # rough rule-of-thumb

# ---------- single experiment ----------------------------------------------
def bench_case(n, p, lam, algo):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 2))
    y = x + 0.2 * rng.normal(size=(n, 2))

    M = construct_cost(x, y, p=p)
    a = rng.random(n); a /= a.sum()
    b = rng.random(n); b /= b.sum()

    if algo == "LP":
        t0 = time.perf_counter()
        cost = exact_ot_cost(M, a, b)
        dt  = 1e3 * (time.perf_counter() - t0)
        return dt, cost, cost, np.nan

    if algo == "Greenkhorn":
        try:
            import ot
        except ImportError:
            return np.nan, np.nan, np.nan, np.nan
        t0 = time.perf_counter()
        # Compute transport matrix γ and extract cost
        gamma = ot.bregman.greenkhorn(a, b, M, reg=1/lam)
        cost = np.sum(gamma * M)  # Compute cost manually
        dt = 1e3 * (time.perf_counter() - t0)
        return dt, cost, exact_ot_cost(M, a, b), np.nan

    # default → Sinkhorn
    t0 = time.perf_counter()
    cost = sinkhorn_distance(M, lam, a, b)
    dt   = 1e3 * (time.perf_counter() - t0)
    flop = flops_sinkhorn(n, n)
    return dt, cost, exact_ot_cost(M, a, b), flop

# ---------- CLI -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default="bench.csv")
    ap.add_argument("--sizes",   default="50,100,200,400")
    ap.add_argument("--lam",     default="20,50,100,200,400")
    ap.add_argument("--p",       default="1,2")
    ap.add_argument("--algos",   default="Sinkhorn,LP")
    
    args = ap.parse_args()

    sizes = [int(s)   for s in args.sizes.split(",")]
    lams  = [int(l)   for l in args.lam.split(",")]
    ps    = [float(p) for p in args.p.split(",")]
    algos = args.algos.split(",")

    with open(args.outfile, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow("size,p,lam,algo,time_ms,cost,exact_cost,flops".split(","))
        for n, p, lam, algo in itertools.product(sizes, ps, lams, algos):
            dt, cost, exact_cost, flop = bench_case(n, p, lam, algo)
            w.writerow([n, p, lam, algo, f"{dt:.2f}", cost, exact_cost, flop])
            print(f"{algo:10s} n={n:<4} p={p} λ={lam:<3}  {dt:7.1f} ms")

if __name__ == "__main__":
    main()

