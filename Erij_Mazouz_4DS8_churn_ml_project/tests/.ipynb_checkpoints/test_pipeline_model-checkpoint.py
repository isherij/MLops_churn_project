# test_model_pipeline.py
import pytest

from model_pipeline import prepare_data, train_model, predict


 
# Test prepare_data function
def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data()
    print("test prepare data")  # Log to Airflow's logger
    assert len(X_train) > 0, "X_train should not be empty"
    assert len(X_test) > 0, "X_test should not be empty"
    assert len(X_train.shape) == 2, "X_train should be 2D"


# Test train function
def test_train_model():
    X_train, _, y_train, _ = prepare_data()
    model = train_model(X_train, y_train)
    print("test prepare data")  # Log to Airflow's logger
    assert model is not None, "Model should not be None"
    assert hasattr(model, "predict"), "Model should have a predict method"


def test_predictions():
    X_train, X_test, y_train, _ = prepare_data()
    model = train_model(X_train, y_train)
    print("test prepare data")  # Log to Airflow's logger
    # Corrected way to call the predict method:
    y_pred = model.predict(
        X_test
    )  # Assuming `model.predict` is how it should be called
    assert len(y_pred) == len(X_test), "Number of predictions should match test data"
