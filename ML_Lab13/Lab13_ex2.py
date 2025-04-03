import random
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Load and preprocess the Diabetes dataset
def load_diabetes_data():
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Standardize features for better training stability
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Split dataset manually
def train_test_split_manual(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# Find the best split for a decision tree
def best_split(X, y):
    best_feature, best_threshold = None, None
    min_error = float("inf")

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask, right_mask = X[:, feature_idx] <= threshold, X[:, feature_idx] > threshold
            left_y, right_y = y[left_mask], y[right_mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue  # Skip invalid splits

            error = np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)

            if error < min_error:
                min_error = error
                best_feature, best_threshold = feature_idx, threshold

    return best_feature, best_threshold

# Train a decision tree recursively
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

# Predict using a trained decision tree
def predict(tree, X):
    if not isinstance(tree, tuple):
        return np.full(len(X), tree) if len(X.shape) > 0 else tree

    feature, threshold, left_subtree, right_subtree = tree
    left_mask, right_mask = X[:, feature] <= threshold, X[:, feature] > threshold

    predictions = np.zeros(X.shape[0])
    predictions[left_mask] = predict(left_subtree, X[left_mask])
    predictions[right_mask] = predict(right_subtree, X[right_mask])

    return predictions

# Bootstrap sampling
def bootstrap_sampling(X, y, max_samples):
    n_samples = int(len(X) * max_samples)
    indices = [random.randint(0, len(X) - 1) for _ in range(n_samples)]
    return X[indices], y[indices]

# Train Bagging Regressor
def train_bagging_regressor(X_train, y_train, n_estimators=10, max_samples=0.8):
    models = []
    for _ in range(n_estimators):
        X_sample, y_sample = bootstrap_sampling(X_train, y_train, max_samples)
        tree = build_tree(X_sample, y_sample, max_depth=5, min_samples_split=2)
        models.append(tree)
    return models

# Predict using Bagging Regressor
def predict_bagging_regressor(models, X_test):
    predictions = np.array([predict(tree, X_test) for tree in models])
    return np.mean(predictions, axis=0)

# Main function
def main():
    # Load and preprocess dataset
    X, y = load_diabetes_data()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)

    # Train the custom bagging regressor
    models = train_bagging_regressor(X_train, y_train, n_estimators=10, max_samples=0.8)

    # Predict and evaluate
    y_pred = predict_bagging_regressor(models, X_test)
    R2 = r2_score(y_test, y_pred)

    print(f"R² Score on Diabetes Dataset: {R2:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
