import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] > self.threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            for feature in range(n_features):
                feature_values = np.unique(X[:, feature])
                for threshold in feature_values:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X[:, feature] < threshold] = -1
                        else:
                            predictions[X[:, feature] > threshold] = -1

                        error = np.sum(w[y != predictions])

                        if error < min_error:
                            min_error = error
                            stump.feature_index = feature
                            stump.threshold = threshold
                            stump.polarity = polarity

            stump.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
            predictions = stump.predict(X)

            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)

            self.stumps.append(stump)
            self.alphas.append(stump.alpha)

    def predict(self, X):
        stump_preds = np.array([stump.alpha * stump.predict(X) for stump in self.stumps])
        return np.sign(np.sum(stump_preds, axis=0))


# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using only two features for visualization purposes
y = iris.target

# Convert to binary classification (class 0 vs. class 1 and 2)
y = np.where(y == 0, -1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# Train AdaBoost
model = AdaBoost(n_estimators=20)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


