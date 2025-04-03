from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_data():
    X,y=load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

    adaboost_classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        random_state=42
    )
    adaboost_classifier.fit(X_train,y_train)

    pred=adaboost_classifier.predict(X_test)
    cm=confusion_matrix(y_test,pred)
    accuracy=accuracy_score(y_test,pred)

    return cm, accuracy

def main():
    cm,accuracy=load_data()

    print("Confusion Matrix:\n",cm)
    print(f"Accuracy score: {accuracy:.4f}")

if __name__ == '__main__':
    main()
