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
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

# Streamlit interface
st.title("Anomaly Detection on Trade Metrics Data")

# Create tabs
tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Modelling"])

# Load the data
file_path = 'reduced_1.csv'
data = pd.read_csv(file_path)

# Check if data loaded correctly
st.write("Data loaded successfully.")
st.write(data.head())

# Separate numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=np.number).columns
non_numeric_cols = data.select_dtypes(exclude=np.number).columns

# Handle missing values with SimpleImputer
imputer_numeric = SimpleImputer(strategy='mean')
imputer_non_numeric = SimpleImputer(strategy='most_frequent')

# Impute numeric columns
if not numeric_cols.empty:
    data_numeric_imputed = pd.DataFrame(imputer_numeric.fit_transform(data[numeric_cols]), columns=numeric_cols)
else:
    data_numeric_imputed = pd.DataFrame()

# Impute non-numeric columns
if not non_numeric_cols.empty:
    data_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(data[non_numeric_cols]), columns=non_numeric_cols)
else:
    data_non_numeric_imputed = pd.DataFrame()

# Combine the imputed numeric and non-numeric data
data_imputed = pd.concat([data_numeric_imputed, data_non_numeric_imputed], axis=1)

# Check if data imputation was successful
st.write("Data after imputation:")
st.write(data_imputed.head())

# Define preprocessing steps
preprocessor = StandardScaler()

# Fit preprocessing on the data
X = data_imputed.select_dtypes(include=np.number)  # Use only numeric columns for modeling

# Check if X is not empty
if X.empty:
    st.error("No numeric data available for preprocessing.")
else:
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Modify the dataset (e.g., shuffling the data)
    np.random.seed(42)  # Fix the random seed for reproducibility
    np.random.shuffle(X_preprocessed)

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

    with tab2:
        st.header("Exploratory Data Analysis")
        
        st.subheader("Data Preview")
        st.write(data.head())
        
        st.subheader("Summary Statistics")
        summary_stats = data.describe().T  # Transpose the summary statistics
        st.write(summary_stats)
        
        st.subheader("Missing Values")
        st.write(data.isnull().sum())
        
        st.subheader("Correlation Matrix")
        correlation_matrix = data.corr()
        
        # Display correlation matrix as a heatmap
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Display correlation matrix as a table
        st.write("Correlation Matrix Values:")
        st.write(correlation_matrix)

        st.subheader("Pair Plot")
        st.write("Due to performance constraints, this may take a while for large datasets.")
        if st.button("Generate Pair Plot"):
            fig = sns.pairplot(data)
            st.pyplot(fig)

    with tab3:
        st.header("Model Accuracy")

        # Note: Accuracy calculation may not be meaningful for unsupervised anomaly detection
        # Consider using another evaluation metric or comparison approach

        # Temporarily select Isolation Forest as the best model for demonstration
        best_model_name = "Isolation Forest"

        st.subheader(f"Best Model: {best_model_name}")

        # Fit the best model on the entire dataset and score the data
        if best_model_name == "Isolation Forest":
            model = iforest
            scores = model.decision_function(X_preprocessed)
            labels = model.predict(X_preprocessed)
        elif best_model_name == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_preprocessed)
            scores = np.ones_like(labels)  # DBSCAN does not have a scoring function
        elif best_model_name == "HDBSCAN":
            model = HDBSCAN(min_cluster_size=5)
            labels = model.fit_predict(X_preprocessed)
            scores = model.outlier_scores_
        elif best_model_name == "KMeans":
            model = KMeans(n_clusters=2, random_state=42)
            labels = model.predict(X_preprocessed)
            scores = -model.transform(X_preprocessed).min(axis=1)  # Inverse distance to cluster center
        elif best_model_name == "Local Outlier Factor":
            model = LocalOutlierFactor(novelty=False, contamination='auto')
            labels = model.fit_predict(X_preprocessed)
            scores = -model.negative_outlier_factor_  # LOF uses negative outlier factor
        elif best_model_name == "One-Class SVM":
            model = OneClassSVM(kernel='rbf', nu=0.05)
            model.fit(X_preprocessed)
            labels = model.predict(X_preprocessed)
            scores = model.decision_function(X_preprocessed)

        # Convert labels to -1 for outliers and 1 for normal points
        if best_model_name in ["Isolation Forest", "One-Class SVM"]:
            labels = np.where(labels == 1, 1, -1)
        else:
            labels = np.where(labels == -1, -1, 1)

        # Add scores and labels to the original data
        data['Score'] = scores
        data['Anomaly_Label'] = labels

        st.subheader(f"Scoring the Input Data Using {best_model_name}")
        st.write(data[['Score', 'Anomaly_Label']])

        st.subheader("Data with Anomaly Labels")
        st.write(data)

        # Count the occurrences of -1 and 1 in the Anomaly_Label column
        count_anomalies = data['Anomaly_Label'].value_counts()
        st.subheader("Anomaly Label Counts")
        st.write(f"Count of -1 (Outliers): {count_anomalies.get(-1, 0)}")
        st.write(f"Count of 1 (Normal): {count_anomalies.get(1, 0)}")
