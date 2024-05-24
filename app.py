import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
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

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        return None

# Function to preprocess data
def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    preprocessor = StandardScaler()
    X = data_imputed.drop(columns=['Outlier']) if 'Outlier' in data_imputed.columns else data_imputed
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

# Function to train models
def train_models(X_train, X_test):
    # Define and fit Isolation Forest
    iforest = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
    iforest.fit(X_train)
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

    return outlier_preds, predictions_dbscan, predictions_hdbscan, predictions_kmeans, predictions_lof, predictions_svm

# Function to calculate accuracies
def calculate_accuracies(outlier_preds, predictions_dbscan, predictions_hdbscan, predictions_kmeans, predictions_lof, predictions_svm):
    accuracy_dbscan = accuracy_score(outlier_preds, predictions_dbscan)
    accuracy_hdbscan = accuracy_score(outlier_preds, predictions_hdbscan)
    accuracy_kmeans = accuracy_score(outlier_preds, predictions_kmeans)
    accuracy_lof = accuracy_score(outlier_preds, predictions_lof)
    accuracy_svm = accuracy_score(outlier_preds, predictions_svm)
    accuracy_iforest = accuracy_score(outlier_preds, outlier_preds)
    return accuracy_dbscan, accuracy_hdbscan, accuracy_kmeans, accuracy_lof, accuracy_svm, accuracy_iforest

# Streamlit App
st.title('Anomaly Detection')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Upload", "EDA", "Modelling"])

# Data Upload Tab
with tab1:
    st.header("Upload Your Data")
    data = load_data()
    if data is not None:
        st.write("Data Loaded Successfully")
        st.write(data.head())

# EDA Tab
with tab2:
    st.header("Exploratory Data Analysis")
    if data is not None:
        st.write("Data Description")
        st.write(data.describe())

        st.write("Pairplot")
        sns.pairplot(data)
        st.pyplot(plt)

        st.write("Correlation Heatmap")
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(plt)
    else:
        st.write("Please upload data in the 'Data Upload' tab")

# Modelling Tab
with tab3:
    st.header("Model Training and Evaluation")
    if data is not None:
        # Preprocess data
        X_preprocessed = preprocess_data(data)

        # Separate the data into training and testing sets
        X_train, X_test, _, _ = train_test_split(X_preprocessed, X_preprocessed, test_size=0.3, random_state=42)

        # Train models and get predictions
        outlier_preds, predictions_dbscan, predictions_hdbscan, predictions_kmeans, predictions_lof, predictions_svm = train_models(X_train, X_test)

        # Calculate accuracies
        accuracy_dbscan, accuracy_hdbscan, accuracy_kmeans, accuracy_lof, accuracy_svm, accuracy_iforest = calculate_accuracies(outlier_preds, predictions_dbscan, predictions_hdbscan, predictions_kmeans, predictions_lof, predictions_svm)

        # Display accuracies
        st.write(f"Accuracy for DBSCAN: {accuracy_dbscan}")
        st.write(f"Accuracy for HDBSCAN: {accuracy_hdbscan}")
        st.write(f"Accuracy for KMeans: {accuracy_kmeans}")
        st.write(f"Accuracy for Local Outlier Factor: {accuracy_lof}")
        st.write(f"Accuracy for One-Class SVM: {accuracy_svm}")
        st.write(f"Accuracy for Isolation Forest: {accuracy_iforest}")
    else:
        st.write("Please upload data in the 'Data Upload' tab")


