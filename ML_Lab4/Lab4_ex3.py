from ML_Lab4.Lab4_ex1 import compute_r2, hypothesis, compute_cost, compute_derivative, gradient_descent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    file_path = "/home/ibab/Machine_Learning/ML_Lab3/simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    print("Dataset:")
    print(data.head())

    data = data.dropna()

    X = data.drop(columns=['disease_score_fluct']).values
    y = data['disease_score_fluct'].values.reshape(-1, 1)

    test_split_index = int(len(data) * (0.70))
    X_train = X[:test_split_index]
    y_train = y[:test_split_index]
    X_test = X[test_split_index:]
    y_test = y[test_split_index:]

    return X_train, y_train, X_test, y_test


def scale_features(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


def add_bias_term(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def R2_value(compute_r2):
    return compute_r2


def h_function(hypothesis):
    return hypothesis


def cost_function(compute_cost):
    return compute_cost


def derivatives(compute_derivative):
    return compute_derivative


def gradients(gradient_descent):
    return gradient_descent


# Normal Equation Implementation
def normal_equation(X, y):
    # X.T @ X is the matrix multiplication of X transpose and X
    # np.linalg.inv() computes the inverse of a matrix
    # X.T @ y is the matrix multiplication of X transpose and y
    return np.linalg.inv(X.T @ X) @ X.T @ y



def main():
    X_train, y_train, X_test, y_test = load_data()

    # Standardize the data
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Add bias term to both X_train_scaled and X_test_scaled
    X_train_scaled = add_bias_term(X_train_scaled)
    X_test_scaled = add_bias_term(X_test_scaled)

    # Initialize parameters for gradient descent
    theta_gd = np.zeros((X_train_scaled.shape[1], 1))  # Start with zeros
    alpha = 0.01  # Learning rate for gradient descent
    num_iters = 1000  # Number of iterations for gradient descent

    # Perform gradient descent
    theta_gd, costs = gradient_descent(X_train_scaled, y_train, theta_gd, alpha, num_iters)

    print()
    # Print the learned parameters from gradient descent
    print("Learned parameters (theta) from Gradient Descent:", theta_gd)
    print()

    # Plot the cost function
    # plt.plot(range(num_iters), costs)
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.title('Cost Function Over Iterations')
    # plt.show()


    print()
    # Evaluate on the test set using Gradient Descent
    y_pred_gd = hypothesis(X_test_scaled, theta_gd)
    print("Test predictions from Gradient Descent:", y_pred_gd.flatten())
    print()
    print("Actual test values:", y_test.flatten())

    print()
    # Compute R-squared value for Gradient Descent
    r2_value_gd = compute_r2(y_test, y_pred_gd)
    print(f"R-squared value from Gradient Descent: {r2_value_gd:.4f}")

    # Compute parameters using Normal Equation
    theta_ne = normal_equation(X_train_scaled, y_train)

    print()
    print("-----------------------------------------------")
    print()

    # Print the learned parameters from Normal Equation
    print("Learned parameters (theta) from Normal Equation:", theta_ne)

    print()
    # Evaluate on the test set using Normal Equation
    y_pred_ne = hypothesis(X_test_scaled, theta_ne)
    print("Test predictions from Normal Equation:", y_pred_ne.flatten())
    print()
    print("Actual test values:", y_test.flatten())

    print()
    # Compute R-squared value for Normal Equation
    r2_value_ne = compute_r2(y_test, y_pred_ne)
    print(f"R-squared value from Normal Equation: {r2_value_ne:.4f}")


if __name__ == "__main__":
    main()

#with scikit learn for normal equation for simulated dataset

#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from ML_Lab3.Lab3_ex2 import hypothesis
# from ML_Lab3.Lab3_ex2 import compute_cost
# from ML_Lab3.Lab3_ex2 import compute_derivative
# from ML_Lab3.Lab3_ex2 import gradient_descent
#
#
#
# def load_data():
#     file_path = "/home/ibab/Machine_Learning/ML_Lab3/simulated_data_multiple_linear_regression_for_ML.csv"
#     data = pd.read_csv(file_path)
#     print("Dataset:")
#     print(data.head())
#
#     X = data.drop(columns=['disease_score_fluct'])
#     y = data['disease_score_fluct'].values.reshape(-1, 1)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#
#     X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
#     return X_scaled, y
#
#
# def add_bias_term(X):
#     return np.hstack((np.ones((X.shape[0], 1)), X))
#
#
# def hypothesis_function(hypothesis):
#     return hypothesis
#
#
# def cost_functions(compute_cost):
#     return compute_cost
#
#
# def derivatives_computation(compute_derivative):
#     return compute_derivative
#
#
# def gradients_calculations(gradient_descent):
#     return gradient_descent
#
# def normal_function_implementation(X,y):
#     return np.linalg.inv(X.T @ X) @ X.T @ y
#
#
# def main():
#     X, y = load_data()
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
#
#     theta = np.random.randn(X_train.shape[1], 1) * 0.1
#     alpha = 0.01
#     num_iters = 1000
#     theta, costs = gradient_descent(X_train, y_train, theta, alpha, num_iters)
#
#     theta_ne = normal_function_implementation(X_train, y_train)
#     print("Learned parameters (theta) from Normal Equation:", theta_ne)
#
#     print(f"Final Cost: {costs[-1]}")
#
#     test_predictions_ne = X_test @ theta_ne
#     print("Test predictions from Normal Equation:", test_predictions_ne.flatten())
#     print("Actual test values:", y_test.flatten())
#
#     model = LinearRegression()
#     model.fit(X_train[:, 1:], y_train)  # Ignore the bias term for sklearn
#     y_pred = model.predict(X_test[:, 1:])  # Ignore the bias term for sklearn
#     r2 = r2_score(y_test, y_pred)
#     print("R2 Score =", r2)
#
#     plt.plot(range(num_iters), costs, color='blue')
#     plt.xlabel('Iterations')
#     plt.ylabel('Cost')
#     plt.title('Cost Function Over Iterations')
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()