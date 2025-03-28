import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    file_path = "housing.csv"
    data = pd.read_csv(file_path)
    print("Dataset:")
    print(data.head())

    data=data.dropna()

    X = data.drop(columns=['ocean_proximity','median_house_value']).values
    y = data['median_house_value'].values.reshape(-1, 1)

    test_split_index = int(len(data) * (1 - 0.30))
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


def compute_r2(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    rss = np.sum((y_true - y_pred) ** 2)  # Residual Sum of Squares
    tss = np.sum((y_true - mean_y_true) ** 2)  # Total Sum of Squares
    r2 = 1 - (rss / tss)  # R-squared
    return r2


def hypothesis(X, theta):
    return np.dot(X, theta)


def compute_cost(X, y, theta):
    predictions = hypothesis(X, theta)
    errors = predictions - y
    cost = (1 / (2 * len(y))) * np.sum(errors**2)
    return cost


def compute_derivative(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    errors = predictions - y
    gradients = (1 / m) * np.dot(X.T, errors)
    return gradients


def gradient_descent(X, y, theta, alpha, num_iters,convergence_threshold=1e-6):
    #theta = np.zeros((X.shape[1], 1))
    costs = []

    for i in range(num_iters):
        gradients = compute_derivative(X, y, theta)
        theta -= alpha * gradients
        cost = compute_cost(X, y, theta)
        costs.append(cost)

        if i > 0 and abs(costs[-2] - costs[-1]) < convergence_threshold:
            print(f"Converged at iteration {i} with cost {cost:.4f}")
            break

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta, costs



def main():
    X_train, y_train, X_test, y_test = load_data()

    # Standardize the data
    X_train_scaled, X_test_scaled= scale_features(X_train, X_test)

    X_train_scaled = add_bias_term(X_train_scaled)
    X_test_scaled = add_bias_term(X_test_scaled)

    # Initialize parameters
    theta = np.zeros((X_train_scaled.shape[1], 1))  # Start with zeros
    alpha = 0.001  # Learning rate
    num_iters = 10000  # Number of iterations

    # Perform gradient descent
    theta, costs = gradient_descent(X_train_scaled, y_train, theta, alpha, num_iters,convergence_threshold=1e-6)

    # Print the final parameters
    print("Learned parameters (theta):", theta)

    # Plot the cost function
    # plt.plot(range(num_iters), costs)
    # plt.xlabel('Iterations')
    # plt.ylabel('Cost')
    # plt.title('Cost Function Over Iterations')
    # plt.show()

    test_predictions = hypothesis(X_test_scaled, theta)
    print("Test predictions:", test_predictions.flatten())
    print("Actual test values:", y_test.flatten())


    # Compute R-squared value
    r2_value = compute_r2(y_test, test_predictions)
    print(f"R-squared value: {r2_value:.4f}")


if __name__ == "__main__":
    main()
