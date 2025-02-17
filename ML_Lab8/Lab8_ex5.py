import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data =  pd.read_csv('/home/ibab/breast-cancer.csv')

# X = data.iloc[:, :-1]
# y =  data.iloc[:,-1]
X = data.iloc[:, :-1].astype(str)
y = data.iloc[:, -1].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=20)
print("Dimension of X before Encoding: ",X_train.shape)
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

oh = OneHotEncoder()
oh.fit(X_train)
X_train = oh.transform(X_train)
print("Dimension of X after Encoding: ",X_train.shape)
X_test = oh.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(y_pred)
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy Score: ",accuracy)