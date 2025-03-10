from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import sys
import os

sys.path.insert(0, "/home/erij/airflow/dags")
from model_pipeline import (
    load_model,
    prepare_data,
    train_model,
    save_model,
    evaluate_model,
    predict,
)  # Your custom functions

# Default arguments for the DAG
default_args = {
    "owner": "Erij",
    "depends_on_past": False,
    "start_date": datetime(2025, 2, 20),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "ml_project_dag",  # DAG name
    description="DAG for orchestrating all ML project steps",
    schedule_interval="@daily",  # Set your schedule, this one runs daily
    start_date=datetime(2025, 2, 20),  # Start date for the DAG
    catchup=False,  # No backfilling for past runs
)


# Task 1: Check code formatting using Black
def check_formatting():
    subprocess.run(
        ["black", "--check", "."], check=True
    )  # Added check=True for proper error handling


format_check_task = PythonOperator(
    task_id="check_formatting",
    python_callable=check_formatting,
    dag=dag,
)


# Task 2: Run tests with pytest
def run_tests():
    subprocess.run(
        ["pytest", "tests/"], check=True
    )  # Added check=True for proper error handling


test_task = PythonOperator(
    task_id="run_tests",
    python_callable=run_tests,
    dag=dag,
)


# Task 3: Prepare Data
def prepare_data_task():
    X_train, X_test, y_train, y_test = prepare_data()  # Prepare the data
    print(
        f"Training data: {X_train.shape[0]} samples, Test data: {X_test.shape[0]} samples"
    )


data_preparation_task = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data_task,
    dag=dag,
)


# Task 4: Train the Model
def train_model_task():
    X_train, _, y_train, _ = prepare_data()  # Prepare the data
    model = train_model(X_train, y_train)  # Train the model
    print(f"Model trained: {model}")


model_training_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model_task,
    dag=dag,
)


# Task 5: Save the Model
def save_model_task():
    model = load_model()  # Load the trained model
    save_model(model)  # Save the trained model
    print("Model saved.")


model_saving_task = PythonOperator(
    task_id="save_model",
    python_callable=save_model_task,
    dag=dag,
)


# Task 6: Evaluate the Model
def evaluate_model_task():
    _, X_test, _, y_test = prepare_data()  # Use the prepared test data
    model = load_model()  # Load the model
    accuracy, class_report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{class_report}")


model_evaluation_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model_task,
    dag=dag,
)


# Task 7: Predict with the Model
def make_predictions_task():
    _, X_test, _, _ = prepare_data()  # Get the test features
    model = load_model()  # Load the trained model
    prediction = predict(X_test)  # Make predictions
    print(f"Predictions: {prediction}")


predict_task = PythonOperator(
    task_id="make_predictions",
    python_callable=make_predictions_task,
    dag=dag,
)


# Task 8: Clean up (optional)
def cleanup_task():
    model_file = "model.joblib"
    if os.path.exists(
        model_file
    ):  # Ensure the model file exists before trying to remove it
        os.remove(model_file)
        print("Model file removed.")
    else:
        print("No model file found to remove.")


cleanup_task = PythonOperator(
    task_id="cleanup",
    python_callable=cleanup_task,
    dag=dag,
)

# Set task dependencies
(
    format_check_task
    >> test_task
    >> data_preparation_task
    >> model_training_task
    >> model_saving_task
    >> model_evaluation_task
    >> predict_task
    >> cleanup_task
)
