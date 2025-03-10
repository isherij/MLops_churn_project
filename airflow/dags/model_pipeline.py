# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow

# from elasticsearch import Elasticsearch
import numpy as np

# import inspec
import csv  # CSV file operations
import seaborn as sns  # Visualizations
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder  # Encoding ordinal features
from sklearn.preprocessing import OneHotEncoder  # Encoding nominal features
from sklearn.feature_selection import SelectKBest, chi2  # Feature selection functions
from sklearn.preprocessing import StandardScaler  # Standrization
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)  # Model evaluation metrics
import logging  # Customizing fetch messages
from imblearn.over_sampling import SMOTE  # Handle dataset imbalance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
from imblearn.over_sampling import ADASYN

# import of GBM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay


def prepare_data(
    data_path_train="churn-bigml-80.csv", data_path_test="churn-bigml-20.csv"
):
    training_dataset = pd.read_csv(data_path_train)
    test_dataset = pd.read_csv(data_path_test)

    threshold = 0.9
    nbr_features = 10
    target_column = "Churn"
    categorical_columns = ["State", "International plan", "Voice mail plan", "Churn"]
    mean_cols = [
        "Total day minutes",
        "Total day charge",
        "Total eve minutes",
        "Total eve charge",
        "Total night minutes",
        "Total night charge",
        "Total intl minutes",
        "Total intl charge",
    ]
    median_cols = [
        "Account length",
        "Total day calls",
        "Total eve calls",
        "Total night calls",
        "Total intl calls",
        "Customer service calls",
    ]

    # Helper function to compute bounds for outliers
    def compute_bounds(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    # Replace outliers with mean
    def Replace_outliers_with_mean(df, column):
        lower_bound, upper_bound = compute_bounds(df, column)
        mean_value = df[column].mean()
        df[column] = df[column].where(
            (df[column] >= lower_bound) & (df[column] <= upper_bound), mean_value
        )
        return df

    # Replace outliers with median
    def Replace_outliers_with_median(df, column):
        lower_bound, upper_bound = compute_bounds(df, column)
        median_value = df[column].median()
        df[column] = df[column].where(
            (df[column] >= lower_bound) & (df[column] <= upper_bound), median_value
        )
        return df

    # Encode categorical features
    def encoding_categorical_features(df, columns):
        label_encoder = LabelEncoder()
        for column in columns:
            df[column] = label_encoder.fit_transform(df[column])

    # Delete correlated features
    def delete_correlated_features(df, threshold):
        correlation_matrix = df.corr()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column
            for column in upper_triangle.columns
            if any(abs(upper_triangle[column]) > threshold)
        ]
        df = df.drop(columns=to_drop)
        return df

    # Select best features
    def select_best_features(X, y, nbr_features):
        test = SelectKBest(score_func=chi2, k=nbr_features)
        fit = test.fit(X, y)
        selected_features = X.columns[fit.get_support()]
        X = fit.transform(X)
        return X, selected_features

    # 1. Handle outliers by replacing with mean or median
    for column in mean_cols:
        training_dataset = Replace_outliers_with_mean(training_dataset, column)
    for column in median_cols:
        training_dataset = Replace_outliers_with_median(training_dataset, column)

    # 2. Remove rows with extreme outliers based on bounds
    numeric_columns = training_dataset.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        lower_bound, upper_bound = compute_bounds(training_dataset, column)
        outliers = (training_dataset[column] < lower_bound) | (
            training_dataset[column] > upper_bound
        )
        training_dataset = training_dataset[~outliers]

    # 3. Encoding categorical features
    encoding_categorical_features(training_dataset, categorical_columns)

    # 4. Deleting highly correlated features
    training_dataset = delete_correlated_features(training_dataset, threshold)

    # 5. Splitting features and target
    X_train = training_dataset.drop(columns=[target_column])
    y_train = training_dataset[target_column]

    # 6. Feature selection
    X_train, selected_features = select_best_features(X_train, y_train, nbr_features)

    # 7. Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 8. Balancing the data using ADASYN
    # adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    # X_train_scaled, y_train = adasyn.fit_resample(X_train_scaled, y_train)

    # Saving the scaler for later use (for test data scaling)
    joblib.dump(scaler, "scaler.joblib")

    # --- Now for the test dataset ---
    # Apply the same preprocessing steps to the test dataset
    # 1. Handle outliers by replacing with mean or median
    for column in mean_cols:
        test_dataset = Replace_outliers_with_mean(test_dataset, column)
    for column in median_cols:
        test_dataset = Replace_outliers_with_median(test_dataset, column)

    # 2. Remove rows with extreme outliers based on bounds
    numeric_columns = test_dataset.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        lower_bound, upper_bound = compute_bounds(test_dataset, column)
        outliers = (test_dataset[column] < lower_bound) | (
            test_dataset[column] > upper_bound
        )
        test_dataset = test_dataset[~outliers]

    # 3. Encoding categorical features
    encoding_categorical_features(test_dataset, categorical_columns)

    # 4. Deleting highly correlated features
    test_dataset = delete_correlated_features(test_dataset, threshold)

    # 5. Splitting features and target for the test dataset
    X_test = test_dataset.drop(columns=[target_column])
    y_test = test_dataset[target_column]

    # 6. Feature selection
    X_test, selected_features = select_best_features(X_test, y_test, nbr_features)

    # 7. Scaling features for the test set using the saved scaler
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# Train model with adjusted parameters (e.g., larger n_neighbors and distance weights)
def train_model(x_train, y_train):
    # Initialize the GBM model (Gradient Boosting Model)
    gbm_model = GradientBoostingClassifier(random_state=42)
    gbm_model.fit(x_train, y_train)

    # Save the trained model
    joblib.dump(gbm_model, "GBM_model.joblib")  # Save gbm_model instead of knn_model
    print(f"GBM Model trained and saved")

    return gbm_model


# Save model
def save_model(model):
    joblib.dump(model, "model.joblib")


# Load model
def load_model():
    return joblib.load("GBM_model.joblib")


# Evaluate model
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Create confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.savefig("confusion_matrix.png")

    # Generate classification report
    class_report = classification_report(y_test, y_pred)

    # Print the evaluation results
    print("Accuracy score:", accuracy * 100)
    print(f"Classification Report:\n{class_report}")
    print("Confusion matrix saved as 'confusion_matrix.png'")

    return accuracy, class_report


# Predict new data
def predict(features):
    print("Prediction function called!")

    # Load model
    model = load_model()

    # Make prediction
    prediction = model.predict(features)
    print("Prediction result:", prediction[0])

    return prediction
