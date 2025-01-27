from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    file_path = "/home/ibab/Machine_Learning/ML_Lab3/simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    print("Dataset preview:")
    print(data.head())

    X = data[['age']].values
    #y = data['disease_score_fluct']
    y = data['disease_score'].values.reshape(-1, 1)

    return X, y


def hypothesis(X_train_scaled, theta):
    return np.dot(X_train_scaled, theta)


def compute_cost(X_train_scaled, y, theta):
    predictions = hypothesis(X_train_scaled, theta)
    errors = predictions - y
    cost = (1 / (2 * len(y))) * np.sum(errors**2)
    return cost


def compute_derivative(X_train_scaled, y, theta):
    m = len(y)
    predictions = hypothesis(X_train_scaled, theta)
    errors = predictions - y
    gradients = (1 / m) * np.dot(X_train_scaled.T, errors)
    return gradients


def gradient_descent(X_train_scaled, y, theta, alpha, num_iters):
    #theta = np.zeros((X.shape[1], 1))
    costs = []

    for i in range(num_iters):
        gradients = compute_derivative(X_train_scaled, y, theta)
        theta -= alpha * gradients
        cost = compute_cost(X_train_scaled, y, theta)
        costs.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta, costs


def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def plot_results(X_test,y_test, y_pred_gd, y_pred_sklearn, y_pred_ne, feature_name='age'):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test,y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred_gd, color='black', label="Gradient Descent Line", linewidth=8)
    plt.plot(X_test, y_pred_sklearn, color='green', label="Sklearn Line", linewidth=2)
    plt.plot(X_test, y_pred_ne, color='orange', label="Normal Equation Line", linewidth=2)
    plt.xlabel('age')
    plt.ylabel('Target(Scaled Disease Score)')
    plt.title(f'Regression Line Comparison: {feature_name}')
    plt.legend()
    plt.show()


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    ##scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
    # X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))


    # train a model
    print("---Training under progress---")
    print("N =%d" % (len(X)))
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_sklearn = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_sklearn)
    print("R2 =", r2)

    theta = np.zeros((X_train_scaled.shape[1], 1))  # Start with zeros
    alpha = 0.01  # Learning rate
    num_iters = 1000  # Number of iterations

    # Perform gradient descent
    theta, costs = gradient_descent(X_train_scaled, y_train, theta, alpha, num_iters)

    # Print the final parameters
    print("Learned parameters (theta):", theta)

    y_pred_gd = hypothesis(X_test_scaled, theta)
    print("Test predictions:", y_pred_gd.flatten())
    print("Actual test values:", y_test.flatten())

    theta_ne = normal_equation(X_train_scaled, y_train)

    # Print the learned parameters from Normal Equation
    print("Learned parameters (theta) from Normal Equation:", theta_ne)

    print()
    # Evaluate on the test set using Normal Equation
    y_pred_ne = hypothesis(X_test_scaled, theta_ne)
    print("Test predictions from Normal Equation:", y_pred_ne.flatten())
    print()
    print("Actual test values:", y_test.flatten())

    plot_results(X_test, y_test, y_pred_gd, y_pred_sklearn, y_pred_ne, feature_name="age")


if __name__ == "__main__":
    main()

