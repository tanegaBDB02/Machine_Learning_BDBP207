import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def data():
    # generating random regression data
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def train_tree(X_train, X_test, y_train, y_test):
    n_estimators = 20 #number of trees
    max_depth = 10 #max depth of tree
    n_samples = X_train.shape[0] #number of training samples

    #training different trees
    models=[]
    for _ in range(n_estimators):
        indices=np.random.choice(n_samples,size=n_samples,replace=True) #with replacement
        X_sample=X_train[indices]
        y_sample=y_train[indices]
        tree=DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X_sample, y_sample)
        models.append(tree)

    pred=np.array([tree.predict(X_test) for tree in models])
    final_pred=np.mean(pred,axis=0)

    R2_score=r2_score(y_test, final_pred)
    print(f"R2_score={R2_score}")

def main():
    X_train, X_test, y_train, y_test = data()
    train_tree(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()



