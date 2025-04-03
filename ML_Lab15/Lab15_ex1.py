from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ISLP import load_data

def data_load():
    weekly = load_data('Weekly')
    X_weekly = weekly.drop(columns=["Direction"])
    y_weekly = weekly["Direction"].apply(lambda x: 1 if x == "Up" else 0)

    X_train, X_test, y_train, y_test = train_test_split(X_weekly, y_weekly, test_size=0.3,random_state=999)

    return X_train, X_test, y_train, y_test

def gradient_boost_classification(X_train, X_test, y_train, y_test):
    gradient_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=999)
    gradient_classifier.fit(X_train,y_train)

    y_pred = gradient_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Classification Accuracy: {accuracy:.4f}")

def main():
    X_train, X_test, y_train, y_test = data_load()
    gradient_boost_classification(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
