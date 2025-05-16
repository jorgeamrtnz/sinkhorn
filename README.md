# Sinkhorn Algorithm for Entropy-Regularized Optimal Transport (18.335 Spring 2025)

This project provides a from-scratch implementation of the Sinkhorn algorithm for efficiently approximating the p-Wasserstein distance between probability measures. It includes two algorithm variants, performance benchmarks against exact solvers and Greenkhorn, and basic validation tests.

## Background

Optimal transport (OT) seeks the minimum-cost transformation between two probability measures. The Wasserstein metric defines this cost, but exact computation is expensive for large supports. Cuturi (2013) introduced an entropic regularization technique, giving the **Sinkhorn distance**, a fast, approximate OT solver with drastically reduced computational cost.

This project implements:
- A **numerically stable** Sinkhorn algorithm
- A direct translation of **Cuturi's original algorithm (Alg. 1)**
- Comparisons with **exact solvers** and **POT's Greenkhorn**
- FLOP profiling and benchmarking
- Sparse image speedup for largeer datasets (e.g. MNIST)