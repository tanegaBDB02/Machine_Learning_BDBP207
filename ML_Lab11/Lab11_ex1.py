import numpy as np
import pandas as pd
import pprint
from sklearn.datasets import load_iris


def load_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    return X, y


def H(y):
    targets, counts = np.unique(y, return_counts=True)
    p = counts / len(y)  # p = probabilities
    return -np.sum(p * np.log2(p)) if len(p) > 0 else 0  # Avoid log(0)


def IG_fun(X, y, feature, threshold):
    left = X[feature] <= threshold
    right = X[feature] > threshold
    y_left, y_right = y[left], y[right]
    H_parent, H_left, H_right = H(y), H(y_left), H(y_right)
    wl = len(y_left) / len(y) if len(y) > 0 else 0  # wl = weight_left
    wr = len(y_right) / len(y) if len(y) > 0 else 0  # wr = weight_right
    EH = wl * H_left + wr * H_right  # (EH = Expected entropy)
    IG = H_parent - EH
    return IG


def build_tree(X, y, min_samples=5, depth=0, max_depth=4):
    if len(y) <= min_samples or depth >= max_depth or H(y) == 0:
        return np.argmax(np.bincount(y.to_numpy())) if len(y) > 0 else 0  # Return most common class or 0
    best_feature, best_threshold = None, None
    max_IG = -float("inf")
    for feature in X.columns:
        split_val = np.unique(X[feature])
        for threshold in split_val:
            IG = IG_fun(X, y, feature, threshold)
            if IG > max_IG:
                max_IG = IG
                best_feature, best_threshold = feature, threshold
    if best_feature is None:
        return np.argmax(np.bincount(y.to_numpy())) if len(y) > 0 else 0
    left = X[best_feature] <= best_threshold
    right = X[best_feature] > best_threshold
    return {"feature": best_feature, "threshold": best_threshold,
            "left": build_tree(X[left], y[left], min_samples, depth + 1, max_depth),
            "right": build_tree(X[right], y[right], min_samples, depth + 1, max_depth), }


def predict(tree, X):
    if isinstance(tree, (int, np.integer)):  # Leaf node
        return tree
    feature, threshold = tree["feature"], tree["threshold"]
    if X[feature] <= threshold:
        return predict(tree["left"], X)
    else:
        return predict(tree["right"], X)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    X, y = load_data()
    df = pd.concat([X, y], axis=1)
    tree = build_tree(X, y)
    print("\nDecision Tree Structure:")
    pprint.pprint(tree)
    y_pred = np.array([predict(tree, X.iloc[i]) for i in range(len(X))])
    acc = accuracy(y.to_numpy(), y_pred)
    print(f"\nDecision Tree Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()