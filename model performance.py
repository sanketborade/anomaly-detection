import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

# Load the data
data = pd.read_csv('reduced_variables.csv')

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define preprocessing steps
preprocessor = StandardScaler()

# Fit preprocessing on the data
X = data_imputed.drop(columns=['Outlier']) if 'Outlier' in data_imputed.columns else data_imputed
X_preprocessed = preprocessor.fit_transform(X)

# Create synthetic outlier labels for evaluation
np.random.seed(42)
outlier_labels = np.random.choice([1, -1], size=X_preprocessed.shape[0], p=[0.9, 0.1])

# Add noise to the dataset
noise = np.random.normal(0, 1, X_preprocessed.shape)
X_preprocessed_noisy = X_preprocessed + noise

# Separate the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_noisy, outlier_labels, test_size=0.3, random_state=42)

# Define and fit Isolation Forest with fewer estimators
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
accuracy_dbscan = accuracy_score(y_test, predictions_dbscan)
accuracy_hdbscan = accuracy_score(y_test, predictions_hdbscan)
accuracy_kmeans = accuracy_score(y_test, predictions_kmeans)
accuracy_lof = accuracy_score(y_test, predictions_lof)
accuracy_svm = accuracy_score(y_test, predictions_svm)

# Calculate accuracy, precision, and recall for Isolation Forest
accuracy_iforest = accuracy_score(y_test, outlier_preds)

print(f"Accuracy for DBSCAN: {accuracy_dbscan}")
print(f"Accuracy for HDBSCAN: {accuracy_hdbscan}")
print(f"Accuracy for KMeans: {accuracy_kmeans}")
print(f"Accuracy for Local Outlier Factor: {accuracy_lof}")
print(f"Accuracy for One-Class SVM: {accuracy_svm}")
print(f"Accuracy for Isolation Forest: {accuracy_iforest}")
