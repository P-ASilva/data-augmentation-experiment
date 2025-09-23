import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    np.random.seed(42)
    samples_per_class = 100
    
    class_params = {
        0: {"mean": [2, 3], "std": [0.8, 2.5]},
        1: {"mean": [5, 6], "std": [1.2, 1.9]},
        2: {"mean": [8, 1], "std": [0.9, 0.9]},
        3: {"mean": [15, 4], "std": [0.5, 2.0]},
    }
    
    X, y = [], []
    for cls, p in class_params.items():
        data = np.random.normal(loc=p["mean"], scale=p["std"], size=(samples_per_class, 2))
        X.append(data)
        y.append(np.full(samples_per_class, cls))

    X = np.vstack(X)
    y = np.hstack(y)
    
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    colors = ["red", "blue", "green", "orange"]
    for cls in class_params.keys():
        plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f"Class {cls}", alpha=0.6)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Synthetic 2D Data for 4 Classes")
    plt.legend()
    plt.grid(True)
    
    scatter_path = assets_dir / "scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "samples_per_class": samples_per_class,
        "num_classes": len(class_params),
        "class_params": class_params,
        "scatter_plot_path": "assets/scatter.png"
    }

if __name__ == "__main__":
    result = main()
    print("Template variables generated successfully")