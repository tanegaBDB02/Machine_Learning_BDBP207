import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def calculate_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            new_centroids[i] = np.mean(points, axis=0)
    return new_centroids

def main():
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

    # Parameters
    k = 3
    max_iters = 100
    tolerance = 1e-4

    # Initialize centroids randomly
    np.random.seed(42)
    random_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[random_indices]

    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = calculate_centroids(X, labels, k)

        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tolerance):
            print(f"Converged after {i+1} iterations")
            break

        centroids = new_centroids

    # Print assigned clusters
    print("Assigned clusters for each point:")
    print(labels)

    # Plot the final clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title("K-Means Clustering (No Class Version)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
