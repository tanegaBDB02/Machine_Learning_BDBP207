import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def best_split(X, y, min_gain=1e-7): #threshold
    m, n = X.shape
    best_feature, best_threshold, best_mse = None, None, float("inf")
    current_mse = np.var(y)

    for feature in range(n):
        sorted_indices = np.argsort(X[:, feature])
        X_sorted, y_sorted = X[sorted_indices, feature], y[sorted_indices]

        for i in range(1, m):
            threshold = (X_sorted[i - 1] + X_sorted[i]) / 2
            left_mask, right_mask = X[:, feature] <= threshold, X[:, feature] > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_mse, right_mse = np.var(y[left_mask]), np.var(y[right_mask])
            weighted_mse = (np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse) / m
            gain = current_mse - weighted_mse

            if gain > min_gain and weighted_mse < best_mse:
                best_feature, best_threshold, best_mse = feature, threshold, weighted_mse

    return best_feature, best_threshold


def build_tree(X, y, depth=0, max_depth=10, min_samples_split=5):
    if len(y) < min_samples_split or depth >= max_depth or np.var(y) == 0:
        return np.mean(y)

    feature, threshold = best_split(X, y)
    if feature is None:
        return np.mean(y)

    left_mask, right_mask = X[:, feature] <= threshold, X[:, feature] > threshold
    left_subtree = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split)

    return (feature, threshold, left_subtree, right_subtree)


def predict(tree, X):
    if not isinstance(tree, tuple):
        return np.full(len(X), tree) if len(X.shape) > 0 else tree

    feature, threshold, left_subtree, right_subtree = tree
    left_mask, right_mask = X[:, feature] <= threshold, X[:, feature] > threshold

    predictions = np.zeros(X.shape[0])
    predictions[left_mask] = predict(left_subtree, X[left_mask])
    predictions[right_mask] = predict(right_subtree, X[right_mask])

    return predictions


def train_and_evaluate(X, y, test_size=0.2, max_depth=10, min_samples_split=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    tree = build_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
    y_pred = predict(tree, X_test)

    r2_score = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    return r2_score


def main():
    data = load_diabetes()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    r2 = train_and_evaluate(X, y, test_size=0.2, max_depth=10, min_samples_split=5)
    print(f"Test R² Score: {r2:.4f}")


if __name__ == "__main__":
    main()
