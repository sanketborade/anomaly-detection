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

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    preprocessor = StandardScaler()
    X = data_imputed.drop(columns=['Outlier']) if 'Outlier' in data_imputed.columns else data_imputed
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed

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

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load and preprocess data
    data = load_data(uploaded_file)
    X_preprocessed = preprocess_data(data)

    # Separate the data into training and testing sets
    X_train, X_test, _, _ = train_test_split(X_preprocessed, X_preprocessed, test_size=0.3, random_state=42)

    # Train models and get predictions
    outlier_preds, predictions_dbscan, predictions_hdbscan, predictions_kmeans, predictions_lof, predictions_svm = train_models(X_train, X_test)

    # Calculate accuracies
    accuracy_dbscan, accuracy_hdbscan, accuracy_kmeans, accuracy_lof, accuracy_svm, accuracy_iforest = calculate_accuracies(outlier_preds, predictions_dbscan, predictions_hdbscan, predictions_kmeans, predictions_lof, predictions_svm)

    # Exploratory Data Analysis (EDA) tab
    st.header("Exploratory Data Analysis")
    st.subheader("Data Overview")
    st.write(data.describe())
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    
    st.subheader("Data Distribution")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        st.write(f"Distribution for {col}")
        fig, ax = plt.subplots()
        sns.histplot(data[col], ax=ax)
        st.pyplot(fig)
    
    st.subheader("Pairplot")
    fig = sns.pairplot(data[numeric_cols])
    st.pyplot(fig)

    # Modelling tab
    st.header("Modelling")
    st.subheader("Accuracy of Models")

    st.write(f"Accuracy for DBSCAN: {accuracy_dbscan}")
    st.write(f"Accuracy for HDBSCAN: {accuracy_hdbscan}")
    st.write(f"Accuracy for KMeans: {accuracy_kmeans}")
    st.write(f"Accuracy for Local Outlier Factor: {accuracy_lof}")
    st.write(f"Accuracy for One-Class SVM: {accuracy_svm}")

    # Determine the model with the highest accuracy
    accuracies = {
        "DBSCAN": accuracy_dbscan,
        "HDBSCAN": accuracy_hdbscan,
        "KMeans": accuracy_kmeans,
        "Local Outlier Factor": accuracy_lof,
        "One-Class SVM": accuracy_svm
    }

    best_model = max(accuracies, key=accuracies.get)
    st.subheader(f"Best Model: {best_model} with accuracy of {accuracies[best_model]:.2f}")

else:
    st.write("Please upload a CSV file to proceed.")
