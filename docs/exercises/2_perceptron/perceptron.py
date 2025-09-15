import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure path code does not implode
os.makedirs("docs/exercises/2_perceptron", exist_ok=True)
os.makedirs("docs/exercises/2_perceptron/img", exist_ok=True)

# ----- Perceptron Implementation -----
def perceptron_train(X, y, eta=0.01, max_epochs=100):
    w = np.zeros(X.shape[1])
    b = 0
    accuracies = []

    for epoch in range(max_epochs):
        errors = 0
        for xi, target in zip(X, y):
            if target * (np.dot(w, xi) + b) <= 0:  # misclassified
                w += eta * target * xi
                b += eta * target
                errors += 1
        acc = np.mean(np.sign(np.dot(X, w) + b) == y)
        accuracies.append(acc)
        if errors == 0:  # convergence
            break
    return w, b, accuracies

# ----- Plotting Functions -----
def plot_boundary(X, y, w, b, title, filename_prefix):
    plt.figure(figsize=(8,6))
    plt.scatter(X[y==-1,0], X[y==-1,1], color="red", label="Class 0")
    plt.scatter(X[y==1,0], X[y==1,1], color="blue", label="Class 1")

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(w[0]*x_vals + b)/w[1]
    plt.plot(x_vals, y_vals, 'k--', label="Decision boundary")

    preds = np.sign(np.dot(X, w) + b)
    misclassified = X[preds != y]
    if len(misclassified) > 0:
        plt.scatter(misclassified[:,0], misclassified[:,1],
                    facecolors='none', edgecolors='k', s=100, label="Misclassified")

    plt.legend()
    plt.title(title)
    plt.savefig(f"docs/exercises/2_perceptron/img/{filename_prefix}_boundary.png")
    plt.close()

    # Accuracy plot
    plt.plot(range(1, len(accuracies)+1), accuracies, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Training Accuracy")
    plt.savefig(f"docs/exercises/2_perceptron/img/{filename_prefix}_accuracy.png")
    plt.close()

def run_test(name, mean0, cov0, mean1, cov1, explanation):
    # Data generation
    class0 = np.random.multivariate_normal(mean0, cov0, 1000)
    class1 = np.random.multivariate_normal(mean1, cov1, 1000)
    X = np.vstack((class0, class1))
    y = np.hstack((-1*np.ones(1000), np.ones(1000)))

    # Training
    global accuracies
    w, b, accuracies = perceptron_train(X, y)

    # Plots
    plot_boundary(X, y, w, b, f"{name} Decision Boundary", name.lower().replace(" ", "_"))

    # Report text
    final_acc = accuracies[-1]
    epochs_used = len(accuracies)

    report = f"""
# {name} Report

**Final Weights:** {w}  
**Final Bias:** {b:.4f}  
**Final Accuracy:** {final_acc*100:.2f}%  
**Epochs until convergence:** {epochs_used}

### Analysis
{explanation}

![Decision Boundary](./img/{name.lower().replace(" ", "_")}_boundary.png)  
![Accuracy Curve](./img/{name.lower().replace(" ", "_")}_accuracy.png)

"""
    return report

# ----- Exercise 1 -----
exercise1_report = run_test(
    "Exercise 1",
    mean0=[1.5, 1.5], cov0=[[0.5, 0],[0,0.5]],
    mean1=[5, 5], cov1=[[0.5, 0],[0,0.5]],
    explanation=(
        "The perceptron converged quickly because the data is linearly separable. "
        "Clusters are compact and far apart, so the decision boundary is learned in few epochs."
    )
)

# ----- Exercise 2 -----
exercise2_report = run_test(
    "Exercise 2",
    mean0=[3, 3], cov0=[[1.5, 0],[0,1.5]],
    mean1=[4, 4], cov1=[[1.5, 0],[0,1.5]],
    explanation=(
        "Here, the means are closer and the variance is higher, causing overlap between classes. "
        "This prevents perfect linear separation, so the perceptron most likely will not converge. "
        "Training may oscillate or plateau, highlighting the model's limitation with non-separable data."
    )
)

# ----- Save Combined Report -----
with open("docs/exercises/2_perceptron/main.md", "w") as f:
    f.write(exercise1_report)
    f.write(exercise2_report)

print("Reports and plots saved under docs/exercises/2_perceptron ✅")