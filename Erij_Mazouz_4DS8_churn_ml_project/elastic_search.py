from elasticsearch import Elasticsearch, RequestsHttpConnection
import mlflow

# Connect to Elasticsearch
es = Elasticsearch(
    hosts=[{"host": "localhost", "port": 9200}],
    connection_class=RequestsHttpConnection,
    headers={"Content-Type": "application/json"},  # Force application/json content-type
)


def log_to_elasticsearch(run_id, experiment_id, params, metrics):
    """
    Logs MLflow run details to Elasticsearch.
    """
    log_data = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "params": params,
        "metrics": metrics,  # Ensure metrics include accuracy and other relevant metrics
    }

    # Send data to Elasticsearch
    try:
        es.index(index="mlflow-logs", body=log_data)
        print(f"Logged run {run_id} with metrics: {metrics}")
    except Exception as e:
        print(f"Error logging to Elasticsearch: {e}")


# MLflow run with accurate logging
with mlflow.start_run() as run:
    # Log parameters to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.01)

    # Simulate or calculate accuracy metric during training
    accuracy_value = 0.93  # Replace this with actual calculated accuracy
    mlflow.log_metric("accuracy", accuracy_value)

    # Fetch the run details after the run starts
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    # Create an MLflow client to fetch the run data
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id).data

    # Extract parameters and metrics
    params = run_data.params
    metrics = run_data.metrics  # This should now be populated correctly

    # Debug: Verify metrics before sending to Elasticsearch
    print("Fetched Metrics from MLflow:", metrics)

    # Send the run details and metrics to Elasticsearch
    log_to_elasticsearch(run_id, experiment_id, params, metrics)

    log_data = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "params": params,
        "metrics": metrics,  # Ensure metrics include accuracy and other relevant metrics
    }

    # Send data to Elasticsearch
    try:
        es.index(index="mlflow-logs", body=log_data)
        print(f"Logged run {run_id} with metrics: {metrics}")
    except Exception as e:
        print(f"Error logging to Elasticsearch: {e}")
