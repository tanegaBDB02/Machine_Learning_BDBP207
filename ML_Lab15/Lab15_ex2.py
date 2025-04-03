from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from ISLP import load_data

def data_load():
    boston = load_data('Boston')
    X_boston = boston.drop(columns=["medv"])
    y_boston = boston["medv"]

    X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.3,random_state=999)

    return X_train, X_test, y_train, y_test

def gradient_boost_regression(X_train, X_test, y_train, y_test):
    gradient_classifier = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=999)
    gradient_classifier.fit(X_train,y_train)

    y_pred = gradient_classifier.predict(X_test)

    R2 = r2_score(y_test, y_pred)
    print(f"Gradient Boosting Regression R2_score: {R2:.4f}")

def main():
    X_train, X_test, y_train, y_test = data_load()
    gradient_boost_regression(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
