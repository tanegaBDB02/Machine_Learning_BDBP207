from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
# print(data.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42 )

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Normal linear regression: \nCoefficients: ",model.coef_)
print("Accuracy:", accuracy_score(y_test, y_pred),"\n")

# A=[0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# for a in A:
#     ridge_clf = RidgeClassifier(alpha=a)
#     ridge_clf.fit(X_train, y_train)
#     y_pred_ridge = ridge_clf.predict(X_test)
#     print("Coefficients: ",ridge_clf.coef_)
#     print("Ridge Classifier Accuracy: ", accuracy_score(y_test, y_pred_ridge), "\n")

A=[0.0001, 0.001, 0.01, 0.1, 1, 10]

for a in A:
    ridge_clf = LogisticRegression(penalty='l2', solver='liblinear', C=a)
    ridge_clf.fit(X_train, y_train)
    y_pred_ridge = ridge_clf.predict(X_test)
    print("Coefficients: ",ridge_clf.coef_)
    print("Ridge Classifier Accuracy: ", accuracy_score(y_test, y_pred_ridge), "\n")

lasso_clf = LogisticRegression(penalty = 'l1', solver= 'liblinear', C=0.1)
lasso_clf.fit(X_train, y_train)
y_pred_lasso = lasso_clf.predict(X_test)
print(lasso_clf.coef_)

print("Lasso Classifier Accuracy:", accuracy_score(y_test, y_pred_lasso))