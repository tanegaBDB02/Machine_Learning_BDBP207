import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#Loading the Iris dataset and selecting only the first two features (sepal length and width)
def load_and_prepare_data(path=None):
    data = load_iris()
    X = data.data[:, :2]  # Only taking SepalLength and SepalWidth
    y = data.target  # Using the encoded labels for the target classes
    return X, y


#Adding a bit of Gaussian noise to the feature values to simulate some variability
def add_noise(X, noise_level=0.2):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    return X + noise


#Taking continuous feature values and turning them into discrete bins
def discretize_features(X, n_bins=6):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_binned = discretizer.fit_transform(X)
    return X_binned.astype(int), discretizer


#Building a joint probability table based on the training data
def build_joint_prob_model(X_train, y_train):
    joint_probs = {}
    label_probs = {}
    unique_labels = np.unique(y_train)

    for label in unique_labels:
        # Filtering out only the rows corresponding to the current label
        label_mask = y_train == label
        label_probs[label] = np.mean(label_mask)
        label_data = X_train[label_mask]

        # Counting how often each combination of feature values appears
        counts = {}
        for row in label_data:
            key = tuple(row)
            counts[key] = counts.get(key, 0) + 1
        total = len(label_data)
        probs = {k: v / total for k, v in counts.items()}
        joint_probs[label] = probs

    return joint_probs, label_probs


#Using the joint probability model to predict the most likely label for each input
def predict_joint_prob(X, joint_probs, label_probs):
    predictions = []

    for row in X:
        best_label = None
        best_prob = -1

        for label in label_probs:
            # Calculating the probability for each label based on the joint probability and prior
            prior = label_probs[label]
            joint = joint_probs[label].get(tuple(row), 1e-6)  # Using a small fallback value for unseen cases
            prob = joint * prior

            if prob > best_prob:
                best_label = label
                best_prob = prob

        predictions.append(best_label)

    return np.array(predictions)


def main():
    #Starting by loading the data and selecting only the first two features
    X, y = load_and_prepare_data()

    #Adding some noise to the data to mimic real-world imperfections
    X_noisy = add_noise(X)

    #Discretizing the noisy features into a fixed number of bins
    X_binned, _ = discretize_features(X_noisy, n_bins=6)

    #Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_binned, y, test_size=0.3, random_state=42)

    #Building the joint probability model using the training data
    joint_probs, label_probs = build_joint_prob_model(X_train, y_train)

    #Making predictions with the joint probability model
    y_pred_joint = predict_joint_prob(X_test, joint_probs, label_probs)

    #Training a decision tree classifier with a maximum depth of 9 for comparison
    tree = DecisionTreeClassifier(max_depth=9, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)

    #Comparing the accuracy of the two models
    acc_joint = accuracy_score(y_test, y_pred_joint)
    acc_tree = accuracy_score(y_test, y_pred_tree)

    print(f"Joint Probability Model Accuracy: {acc_joint:.4f}")
    print(f"Decision Tree (max_depth=2) Accuracy: {acc_tree:.4f}")


if __name__ == "__main__":
    main()
