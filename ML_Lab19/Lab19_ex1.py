import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

def load_data():
    data=pd.read_csv("/home/ibab/Machine_Learning/heart.csv")
    X=data.drop(columns="output")
    y=data["output"]
    return train_test_split(X, y, test_size=0.3, random_state=999)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def model_training(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def metrics(threshold, y_true, y_prob):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Confusion Matrix:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")

    plot_conf_matrix(y_true, y_pred, threshold)

def plot_conf_matrix(y_true, y_pred, threshold):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test = scale_data(X_train, X_test)

    model = model_training(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    for threshold in [0.3, 0.5, 0.7]:
        metrics(threshold, y_test, y_probs)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)

    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()