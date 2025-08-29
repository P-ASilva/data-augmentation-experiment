import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

params = {
    0: {"mean": [2, 3], "std": [0.8, 2.5]},
    1: {"mean": [5, 6], "std": [1.2, 1.9]},
    2: {"mean": [8, 1], "std": [0.9, 0.9]},
    3: {"mean": [15, 4], "std": [0.5, 2.0]},
}

X, y = [], []
for cls, p in params.items():
    data = np.random.normal(loc=p["mean"], scale=p["std"], size=(100, 2))
    X.append(data)
    y.append(np.full(100, cls))

X = np.vstack(X)
y = np.hstack(y)

colors = ["red", "blue", "green", "orange"]
for cls in params.keys():
    plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f"Class {cls}", alpha=0.6)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Synthetic 2D Data for 4 Classes")
plt.legend()
plt.grid(True)
plt.savefig("docs/solutions/ex1/scatter.png", dpi=300)