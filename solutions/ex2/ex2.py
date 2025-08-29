import numpy as np
import matplotlib.pyplot as plt

def pca_2d(X):
    """Return a 2D PCA projection (manual PCA with NumPy)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    W = vecs[:, idx][:, :2]
    return Xc @ W

n_per_class=500
np.random.seed(7)

# --- Class A parameters ---
mu_A = np.zeros(5)
Sigma_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

# --- Class B parameters ---
mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
Sigma_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

XA = np.random.multivariate_normal(mu_A, Sigma_A, size=n_per_class)
XB = np.random.multivariate_normal(mu_B, Sigma_B, size=n_per_class)
X = np.vstack([XA, XB])
y = np.array(["A"] * n_per_class + ["B"] * n_per_class)

X2 = pca_2d(X)

plt.figure()
for label in ["A", "B"]:
    mask = (y == label)
    plt.scatter(X2[mask, 0], X2[mask, 1], label=f"Class {label}", alpha=0.6, s=18)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Exercise 2: PCA (5D → 2D) Scatter")
plt.legend()
plt.grid(True)
plt.savefig("docs/solutions/ex2/pca_scatter.png", dpi=300, bbox_inches="tight")
