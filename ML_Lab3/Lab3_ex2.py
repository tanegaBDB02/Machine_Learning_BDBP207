import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def load_data():
    file_path = "simulated_data_multiple_linear_regression_for_ML.csv"
    data = pd.read_csv(file_path)
    print("Dataset:")
    print(data.head())

    scaler = StandardScaler()
    X= data.drop(columns=['disease_score'])
    X_scaled = scaler.fit_transform(X)
    y = data['disease_score'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)


    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    X_scaled= np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    return X_scaled, y

def hypothesis(X_test, theta):
    return np.dot(X_test, theta)


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


def gradient_descent(X, y, theta, alpha, num_iters):
    #theta = np.zeros((X.shape[1], 1))
    costs = []

    for i in range(num_iters):
        gradients = compute_derivative(X, y, theta)
        theta -= alpha * gradients
        cost = compute_cost(X, y, theta)
        costs.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta, costs


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)

    #theta = np.zeros((X_train.shape[1], 1))
    theta = np.random.randn(X_train.shape[1], 1) * 0.1
    alpha = 0.01
    num_iters = 1000
    theta, costs = gradient_descent(X_train, y_train, theta, alpha, num_iters)
    print(f"Theta: {theta}")
    print(f"Costs: {costs[-1]}")
    # print("Predictions:", hypothesis(X, theta))
    # print("Cost:", compute_cost(X, y, theta))
    print("Derivative:", compute_derivative(X, y, theta))

    plt.plot(range(num_iters), costs, color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Iterations')
    plt.show()


if __name__ == "__main__":
    main()

