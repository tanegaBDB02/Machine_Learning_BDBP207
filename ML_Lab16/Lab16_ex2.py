from xgboost import XGBClassifier, XGBRegressor
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

def load_classifer_data():
    weekly = load_data('Weekly')
    X_weekly = weekly.drop(columns=["Direction"])
    y_weekly = weekly["Direction"].apply(lambda x: 1 if x == "Up" else 0)

    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_weekly, y_weekly, test_size=0.3, random_state=999)

    return X_cls_train, X_cls_test, y_cls_train, y_cls_test

def load_regressor_data():
    boston = load_data('Boston')
    X_boston = boston.drop(columns=["medv"])
    y_boston = boston["medv"]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_boston, y_boston, test_size=0.3, random_state=42)

    return X_reg_train, X_reg_test, y_reg_train, y_reg_test

def XGBoost_classifier(X_cls_train, X_cls_test, y_cls_train, y_cls_test):
    xgb_nodel=XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,random_state=999)
    xgb_nodel.fit(X_cls_train, y_cls_train)

    y_pred=xgb_nodel.predict(X_cls_test)
    accuracy=accuracy_score(y_cls_test, y_pred)

    print(f"\n XGBoost Classifier Results:")
    print(f"Accuracy: {accuracy:.4f}")

def XGBoost_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test):
    xgb_model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_reg_train, y_reg_train)

    y_pred = xgb_model.predict(X_reg_test)
    R2_score = r2_score(y_reg_test, y_pred)
    mse_score = mean_squared_error(y_reg_test, y_pred)

    print(f"\n XGBoost Regressor Results:")
    print(f"R² Score: {R2_score:.4f}")
    print(f"MSE Score: {mse_score:.4f}")

def main():
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = load_classifer_data()
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = load_regressor_data()

    XGBoost_classifier(X_cls_train, X_cls_test, y_cls_train, y_cls_test)
    XGBoost_regressor(X_reg_train, X_reg_test, y_reg_train, y_reg_test)

if __name__ == "__main__":
    main()


