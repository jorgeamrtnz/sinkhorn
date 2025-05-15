import numpy as np
from sklearn.datasets import fetch_openml
from sinkhorn.utils import construct_cost
from sinkhorn.sinkhorn import sinkhorn_distance

def load_pair(digit_a=1, digit_b=1, idx_a=0, idx_b=0):
    """Return two normalised 28x28 histograms for chosen digits."""
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data  = mnist["data"].astype(float) / 255.0
    labels = mnist["target"].astype(int)

    img_a = data[labels == digit_a][idx_a]
    img_b = data[labels == digit_b][idx_b]

    # Each image is length 784; treat pixel intensities as masses
    a = img_a / img_a.sum()
    b = img_b / img_b.sum()

    # Pixel coordinate grid (row, col)
    grid = np.stack(np.divmod(np.arange(784), 28), axis=1)  # (784,2)

    return grid, a, b

def main():
    grid, a, b = load_pair()

    # Cost matrix: squared Euclidean in 2-D
    M = construct_cost(grid, grid, p=2)   # 784×784  (about 5 MB float64)

    C_sink = sinkhorn_distance(M=M, λ=50.0, r=a, C=b)
    print(f"Sinkhorn OT² MNIST-(3→8) ≈ {C_sink:.3f}")

if __name__ == "__main__":
    main()
