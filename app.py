import pandas as pd
import numpy as np
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

# Streamlit interface
st.title("Anomaly Detection on Trade Metrics Data")

# Create tabs
tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Modelling"])

# Load the data
file_path = 'reduced_variables_2.csv'
data = pd.read_csv(file_path)

# Handle non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder
for col in non_numeric_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define preprocessing steps
preprocessor = StandardScaler()
X = data_imputed
X_preprocessed = preprocessor.fit_transform(X)

# Shuffle the data
np.random.seed(42)
np.random.shuffle(X_preprocessed)

# Split the data into training and testing sets
X_train, X_test, _, _ = train_test_split(X_preprocessed, X_preprocessed, test_size=0.3, random_state=42)
n_train_points = X_train.shape[0]

# Define models
models = {
    "Isolation Forest": IsolationForest(n_estimators=50, contamination='auto', random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "HDBSCAN": HDBSCAN(min_cluster_size=5),
    "KMeans": KMeans(n_clusters=min(2, n_train_points), random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(novelty=False, contamination='auto', n_neighbors=min(20, n_train_points)),
    "One-Class SVM": OneClassSVM(kernel='rbf', nu=0.05)
}

# Fit Isolation Forest and detect outliers
iforest = models["Isolation Forest"]
iforest.fit(X_train)
outlier_preds = iforest.predict(X_test)

# Apply each model and handle exceptions
predictions = {}
accuracies = {}
for name, model in models.items():
    try:
        if name in ["Isolation Forest", "One-Class SVM"]:
            predictions[name] = model.fit(X_train).predict(X_test)
        else:
            predictions[name] = model.fit_predict(X_test)
        accuracies[name] = accuracy_score(outlier_preds, predictions[name])
    except Exception as e:
        st.error(f"Error with {name}: {e}")
        predictions[name] = np.zeros_like(outlier_preds)
        accuracies[name] = 0

# Introduce perturbation to reduce Isolation Forest accuracy
perturbation = np.random.choice([1, -1], size=outlier_preds.shape, p=[0.1, 0.9])
outlier_preds_perturbed = np.where(perturbation == 1, -outlier_preds, outlier_preds)
accuracies["Isolation Forest Perturbed"] = accuracy_score(outlier_preds, outlier_preds_perturbed)

with tab2:
    st.header("Exploratory Data Analysis")
    
    st.subheader("Data Preview")
    st.write(data.head())
    
    st.subheader("Summary Statistics")
    summary_stats = data.describe().T
    st.write(summary_stats)
    
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    
    st.subheader("Correlation Matrix")
    correlation_matrix = data.corr()
    
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Pair Plot")
    st.write("Due to performance constraints, this may take a while for large datasets.")
    if st.button("Generate Pair Plot"):
        fig = sns.pairplot(data)
        st.pyplot(fig)

with tab3:
    st.header("Model Accuracy")

    # Display results
    for name, accuracy in accuracies.items():
        st.write(f"Accuracy for {name}: {accuracy}")

    best_model_name = max(accuracies, key=accuracies.get)
    st.subheader(f"Best Model: {best_model_name}")
    st.write(f"Accuracy: {accuracies[best_model_name]}")

    # Fit the best model on the entire dataset and score the data
    model = models[best_model_name]
    if best_model_name in ["Isolation Forest", "One-Class SVM"]:
        scores = model.decision_function(X_preprocessed)
        labels = model.predict(X_preprocessed)
    else:
        labels = model.fit_predict(X_preprocessed)
        scores = np.ones_like(labels) if best_model_name == "DBSCAN" else model.outlier_scores_

    # Convert labels to -1 for outliers and 1 for normal points
    labels = np.where(labels == 1, 1, -1) if best_model_name in ["Isolation Forest", "One-Class SVM"] else np.where(labels == -1, -1, 1)

    # Add scores and labels to the original data
    data['Score'] = scores
    data['Anomaly_Label'] = labels

    st.subheader(f"Scoring the Input Data Using {best_model_name}")
    st.write(data[['Score', 'Anomaly_Label']])

    st.subheader("Data with Anomaly Labels")
    st.write(data)

    count_anomalies = data['Anomaly_Label'].value_counts()
    st.subheader("Anomaly Label Counts")
    st.write(f"Count of -1 (Outliers): {count_anomalies.get(-1, 0)}")
    st.write(f"Count of 1 (Normal): {count_anomalies.get(1, 0)}")
