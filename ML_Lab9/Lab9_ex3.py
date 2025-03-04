import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = "/home/ibab/Machine_Learning/breast_cancer_data.csv"
data= pd.read_csv(file_path)

data = data.drop(columns=['id', "Unnamed: 32"])
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
X = data.drop(columns=['diagnosis'])
y_class = data['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.30, random_state=999)

depth_val=[3,5,10,None]
#none to test a fully grown tree
acc_scores=[]
reports=[]

for depth in depth_val:
    classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    acc_scores.append(accuracy)

    print(f"max_depth={depth}: Accuracy:{accuracy:.4f}")
    print("-----------------")

best_depth = depth_val[acc_scores.index(max(acc_scores))]
print(f"\nBest max_depth: {best_depth}")
