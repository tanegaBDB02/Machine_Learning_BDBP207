from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def load_data(dataset_name):
    if dataset_name == "iris":
        return load_iris(return_X_y=True)
    elif dataset_name == "diabetes":
        return load_diabetes(return_X_y=True)
    else:
        raise ValueError("Invalid dataset name. Choose 'iris' or 'diabetes'.")


def train_bagging_classifier(X_train, y_train):
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(), #weal learner
        n_estimators=10,  # Number of base models
        max_samples=0.8,  # Fraction of training data per model
        max_features=0.8,  # Fraction of features per model
        bootstrap=True, #sampling with replacement
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def train_bagging_regressor(X_train, y_train):
    reg = BaggingRegressor(
        estimator=DecisionTreeRegressor(), #weak learner
        n_estimators=10,  # Number of base models
        max_samples=0.8,  # Fraction of training data per model
        max_features=0.8,  # Fraction of features per model
        bootstrap=True, #sampling with replacement
        random_state=42
    )
    reg.fit(X_train, y_train)
    return reg


def main():
    X, y = load_data("iris")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = train_bagging_classifier(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f'Bagging Classifier Accuracy on Iris dataset: {accuracy_score(y_test, y_pred):.2f}')

    X, y = load_data("diabetes")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = train_bagging_regressor(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(f'Bagging Regressor R2 score on Diabetes dataset: {r2_score(y_test, y_pred):.2f}')


if __name__ == "__main__":
    main()




