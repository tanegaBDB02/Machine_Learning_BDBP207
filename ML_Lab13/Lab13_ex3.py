from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes, load_iris


def train_random_forest_regressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest Regression R2 score: {r2:.2f}")


def train_random_forest_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Classification Accuracy: {accuracy:.2f}")


def main():
    diabetes = load_diabetes()
    train_random_forest_regressor(diabetes.data, diabetes.target)

    iris = load_iris()
    train_random_forest_classifier(iris.data, iris.target)


if __name__ == "__main__":
    main()

