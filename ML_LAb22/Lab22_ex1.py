import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from ISLP import load_data

def data_loading():
    # Load NCI60 data
    NCI60 = load_data('NCI60')
    data = NCI60['data']
    labels = NCI60['labels'].values.ravel()
    return data, labels

def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def reduce_by_clusters(data_matrix, gene_clusters):
    reduced_data = []
    for cluster_id in np.unique(gene_clusters):
        cluster_genes = data_matrix[:, gene_clusters == cluster_id]
        reduced_data.append(cluster_genes.mean(axis=1))
    return np.array(reduced_data).T

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name=""):
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)

    # Support Vector Classifier
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)

    # Evaluation
    print(f"\n--- {model_name} ---")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
    print(f"SVC Accuracy: {accuracy_score(y_test, y_pred_svc):.4f}")

    print("\nConfusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_test, y_pred_log))

    print("\nClassification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_log, zero_division=0))

    return y_pred_log, y_pred_svc

def main():
    # Step 1: Load and standardize data
    data, labels = data_loading()
    data_scaled = standardize_data(data)

    # Step 2: Split data BEFORE PCA or clustering
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data_scaled, labels, test_size=0.3, random_state=42
    )

    # Step 3: PCA reduction
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train_raw)
    X_test_pca = pca.transform(X_test_raw)

    # Step 4: Hierarchical clustering of genes (columns)
    clustering = AgglomerativeClustering(n_clusters=50)
    gene_clusters = clustering.fit_predict(X_train_raw.T)

    # Step 5: Cluster-based dimensionality reduction
    X_train_hier = reduce_by_clusters(X_train_raw, gene_clusters)
    X_test_hier = reduce_by_clusters(X_test_raw, gene_clusters)

    # Step 6: Train and evaluate models
    train_and_evaluate(X_train_hier, X_test_hier, y_train, y_test, model_name="Hierarchical Clustering Features")
    train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test, model_name="PCA Features")

if __name__ == "__main__":
    main()
