import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

# Streamlit interface
st.title("Anomaly Detection")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)

    # Handle missing values with SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Define preprocessing steps
    preprocessor = StandardScaler()

    # Fit preprocessing on the data
    X = data_imputed.drop(columns=['Outlier']) if 'Outlier' in data_imputed.columns else data_imputed
    X_preprocessed = preprocessor.fit_transform(X)

    # Separate the data into training and testing sets
    X_train, X_test, _, _ = train_test_split(X_preprocessed, X_preprocessed, test_size=0.3, random_state=42)

    # Define and fit Isolation Forest
    iforest = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
    iforest.fit(X_train)

    # Detect outliers using Isolation Forest
    outlier_preds = iforest.predict(X_test)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    predictions_dbscan = dbscan.fit_predict(X_test)

    # Apply HDBSCAN
    hdbscan = HDBSCAN(min_cluster_size=5)
    predictions_hdbscan = hdbscan.fit_predict(X_test)

    # Apply KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    predictions_kmeans = kmeans.fit_predict(X_test)

    # Apply Local Outlier Factor (LOF) with novelty=False
    lof = LocalOutlierFactor(novelty=False, contamination='auto')
    predictions_lof = lof.fit_predict(X_test)

    # Apply One-Class SVM
    svm = OneClassSVM(kernel='rbf', nu=0.05)
    predictions_svm = svm.fit_predict(X_test)

    # Calculate accuracy for DBSCAN, HDBSCAN, KMeans, LOF, and One-Class SVM
    accuracy_dbscan = accuracy_score(outlier_preds, predictions_dbscan)
    accuracy_hdbscan = accuracy_score(outlier_preds, predictions_hdbscan)
    accuracy_kmeans = accuracy_score(outlier_preds, predictions_kmeans)
    accuracy_lof = accuracy_score(outlier_preds, predictions_lof)
    accuracy_svm = accuracy_score(outlier_preds, predictions_svm)

    # Create a dictionary to store accuracies
    accuracies = {
        "DBSCAN": accuracy_dbscan,
        "HDBSCAN": accuracy_hdbscan,
        "KMeans": accuracy_kmeans,
        "Local Outlier Factor": accuracy_lof,
        "One-Class SVM": accuracy_svm
    }

    # Select the model with the highest accuracy
    best_model = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model]

    # Label the scores as 1 and -1 based on the best model
    if best_model == "DBSCAN":
        scores = predictions_dbscan
    elif best_model == "HDBSCAN":
        scores = predictions_hdbscan
    elif best_model == "KMeans":
        scores = predictions_kmeans
    elif best_model == "Local Outlier Factor":
        scores = predictions_lof
    elif best_model == "One-Class SVM":
        scores = predictions_svm

    # Labeling scores as 1 and -1
    labels = np.where(scores == -1, -1, 1)

    # Display the best model and accuracy
    st.write(f"Best Model: {best_model}")
    st.write(f"Accuracy: {best_accuracy}")

    # Display the labeled scores
    st.write("Labeled Scores:")
    st.write(labels)
