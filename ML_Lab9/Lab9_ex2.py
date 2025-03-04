import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

file_path = "/home/ibab/Machine_Learning/simulated_data_multiple_linear_regression_for_ML.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=['disease_score', 'disease_score_fluct'])
y = data['disease_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

depth_val=[3,5,10,None]
#none to test a fully grown tree
mse_scores=[]
r2_scores=[]

for depth in depth_val:
    regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse= mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

    print(f"max_depth={depth}: MSE={mse:.4f}, R²={r2:.4f}")

print()
best_depth = depth_val[mse_scores.index(min(mse_scores))]
print(f"Best max_depth: {best_depth}")
