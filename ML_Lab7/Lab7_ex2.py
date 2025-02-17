from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


data_path = "/home/ibab/Machine_Learning/sonarall-data.csv"
data = pd.read_csv(data_path, header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = y.map({'R': 0, 'M': 1})

k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=10000)
scores = []

for fold, (train_indices, test_indices) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fold_score = np.mean(y_pred == y_test)
    scores.append(fold_score)

    print(f"Fold {fold}:")
    print(f"  Training Data Shape: {X_train.shape}")
    print(f"  Test Data Shape: {X_test.shape}")
    print(f"  Accuracy: {round(fold_score, 2)}\n")

mean_accuracy = np.mean(scores)
print(f"Average Accuracy: {mean_accuracy:4f}")
